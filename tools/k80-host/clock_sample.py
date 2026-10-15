#!/usr/bin/env python3
"""Record each die's SM clock, utilization, power, temperature and throttle
reasons while something runs, so a timing can say what clock it ran at (#93).

Sampling only: it reads NVML and never sets a clock or power limit.

  clock_sample.py [--dies 0,1] [--interval 0.05] [--csv trace.csv] -- CMD...
      runs CMD and samples until it exits (CMD's exit code is returned)
  clock_sample.py [--dies ...] [--csv ...]
      samples until Ctrl-C

From Python, around a section to measure:

  from clock_sample import Sampler
  with Sampler(dies=[0]) as s:
      ...
  print(s.summary())

The summary gives, per die: the share of samples at each SM clock, mean
utilization, mean and peak power, peak temperature, the throttle reasons
seen, and how many clock changes the driver logged meanwhile
(nvmlDeviceGetSamples, its own record of the last 100 changes).
"""
import argparse
import collections
import csv
import os
import signal
import subprocess
import sys
import threading
import time

try:
    from vllm.third_party import pynvml as nv
except ImportError:  # outside the vLLM venv
    import pynvml as nv

REASONS = [(0x1, "idle"), (0x2, "app-clock"), (0x4, "sw-power-cap"),
           (0x8, "hw-slowdown"), (0x20, "sw-thermal"), (0x40, "hw-thermal"),
           (0x80, "power-brake")]


def reason_names(bits):
    return [name for bit, name in REASONS if bits & bit] or ["none"]


class Sampler:
    """Samples the given dies (NVML indexes, PCI bus order) in a thread."""

    def __init__(self, dies=None, interval=0.05):
        nv.nvmlInit()
        n = nv.nvmlDeviceGetCount()
        self.dies = list(range(n)) if dies is None else list(dies)
        self.handles = {d: nv.nvmlDeviceGetHandleByIndex(d) for d in self.dies}
        self.interval = interval
        self.rows = []  # (t, die, clock MHz, util %, power W, temp C, reasons)
        self._stop = threading.Event()
        self._thread = None
        self._log_from = {}

    def _clock_log_last(self, h):
        try:
            _, samples = nv.nvmlDeviceGetSamples(h, nv.NVML_PROCESSOR_CLK_SAMPLES, 0)
            return max((s.timeStamp for s in samples), default=0)
        except nv.NVMLError:
            return 0

    def clock_changes(self, d):
        """Clock changes the driver logged since start(): [(us, MHz)]."""
        try:
            _, samples = nv.nvmlDeviceGetSamples(self.handles[d],
                                                 nv.NVML_PROCESSOR_CLK_SAMPLES,
                                                 self._log_from.get(d, 0))
        except nv.NVMLError:
            return []
        return [(s.timeStamp, s.sampleValue.uiVal) for s in samples
                if s.timeStamp > self._log_from.get(d, 0)]

    def _run(self):
        t0 = time.time()
        while not self._stop.is_set():
            t = time.time() - t0
            for d, h in self.handles.items():
                self.rows.append((
                    round(t, 3), d,
                    nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM),
                    nv.nvmlDeviceGetUtilizationRates(h).gpu,
                    nv.nvmlDeviceGetPowerUsage(h) / 1000,
                    nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU),
                    nv.nvmlDeviceGetCurrentClocksThrottleReasons(h)))
            self._stop.wait(self.interval)

    def start(self):
        self._log_from = {d: self._clock_log_last(h)
                          for d, h in self.handles.items()}
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    def summary(self):
        lines = []
        for d in self.dies:
            rows = [r for r in self.rows if r[1] == d]
            if not rows:
                continue
            clocks = collections.Counter(r[2] for r in rows)
            share = ", ".join(f"{c} MHz {n * 100 // len(rows)}%"
                              for c, n in sorted(clocks.items(),
                                                 key=lambda kv: -kv[1])[:4])
            bits = 0
            for r in rows:
                bits |= r[6]
            lines.append(
                f"die {d}: {len(rows)} samples | clock {share} | util "
                f"{sum(r[3] for r in rows) / len(rows):.0f}% | power mean "
                f"{sum(r[4] for r in rows) / len(rows):.0f} W, peak "
                f"{max(r[4] for r in rows):.0f} W | temp peak "
                f"{max(r[5] for r in rows)} C | throttle "
                f"{'+'.join(reason_names(bits))} | driver clock changes "
                f"{len(self.clock_changes(d))}")
        return "\n".join(lines)

    def write_csv(self, path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "die", "sm_mhz", "util_pct", "power_w",
                        "temp_c", "throttle"])
            for r in self.rows:
                w.writerow([*r[:6], "+".join(reason_names(r[6]))])


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dies", help="NVML indexes, comma separated (default all)")
    p.add_argument("--interval", type=float, default=0.05,
                   help="seconds between samples (default 0.05)")
    p.add_argument("--csv", help="write every sample to this file")
    p.add_argument("cmd", nargs=argparse.REMAINDER,
                   help="command to run while sampling (after --)")
    a = p.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    dies = [int(d) for d in a.dies.split(",")] if a.dies else None
    s = Sampler(dies, a.interval).start()
    code = 0
    try:
        if cmd:
            code = subprocess.call(cmd)
        else:
            signal.pause()
    except KeyboardInterrupt:
        pass
    finally:
        s.stop()
        if a.csv:
            s.write_csv(a.csv)
        print(s.summary(), file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
