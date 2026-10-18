/**
 * The dies under test, read while the test runs, so each timed option says what
 * clock and temperature it ran at (#115). A die sets its own clock: the driver
 * lowers it at the 149 W cap, and the hardware slows a die to about 400 MHz at
 * 88 °C, so a tok/s from a hot die measures the throttle, not the server.
 *
 *   const dies = watchDies(2, { startC: 60, stopC: 80 })   nvidia-smi -lms 100, dies 0-1
 *   const gate = await dies.cool()                         waits until every die ≤ startC
 *   ... the option's calls, from → to ...
 *   dies.state(from, to, gate)                             per die: clock, °C, W, flags
 *     └─ invalid when a die reached stopC or raised a fault flag in the window
 *   dies.close()
 *
 * Reads only: nvidia-smi is asked, never set; no clock, power limit or
 * application clock changes.
 *
 * Why 60 °C to start: a die that holds a loaded server sits at 562 MHz and
 * 59-73 W with no request, and levels off at 51-55 °C (measured on #115); it
 * never reaches the 35-43 °C of a die with nothing loaded. 80 °C to stop
 * leaves 8 °C before the 88 °C slowdown, about 20 s at the fastest rise seen.
 */
import { spawn, type ChildProcess } from 'node:child_process';
import { createInterface } from 'node:readline';

/** Throttle reasons as nvidia-smi reports them (clocks_throttle_reasons.active). */
const REASONS: [number, string][] = [
  [0x1, 'idle'], [0x2, 'app-clock'], [0x4, 'sw-power-cap'], [0x8, 'hw-slowdown'],
  [0x20, 'sw-thermal'], [0x40, 'hw-thermal'], [0x80, 'power-brake'],
];
/** The reasons that mean the die, not the server, set the speed. */
const FAULT = 0x4 | 0x8 | 0x20 | 0x40 | 0x80;
const names = (bits: number): string[] => REASONS.filter(([b]) => bits & b).map(([, n]) => n);

/** "GPU0" or "GPU0-3". */
export const dieNames = (dies: number[]): string =>
  dies.length > 1 ? `GPU${dies[0]}-${dies[dies.length - 1]}` : `GPU${dies[0]}`;

export interface DieRule {
  /** Dies 0 to count - 1: the ones the server takes, GPU0 first. */
  dies: number[];
  /** An option starts once every die reads this many °C or less. */
  startC: number;
  /** A die at this many °C makes the option invalid. */
  stopC: number;
}

export interface Gate {
  waitedS: number;
  /** Each die's °C when the option started. */
  startC: Record<number, number>;
}

export interface DieState {
  valid: boolean;
  /** Why the option is invalid; empty when valid. */
  reason: string;
  /** Seconds the option waited for the dies to cool before it started. */
  cooledS: number;
  dies: { die: number; startC: number; medianMhz: number; minMhz: number; peakC: number; meanW: number; flags: string[] }[];
}

interface Sample {
  t: number;
  die: number;
  c: number;
  mhz: number;
  w: number;
  bits: number;
}

/** "2026/10/16 09:46:46.440" in local time, as nvidia-smi writes it, to epoch ms. */
function epoch(stamp: string): number {
  const m = /^(\d+)\/(\d+)\/(\d+) (\d+):(\d+):(\d+)\.(\d+)$/.exec(stamp.trim());
  if (!m) return NaN;
  const [y, mo, d, h, mi, s, ms] = m.slice(1).map(Number);
  return new Date(y, mo - 1, d, h, mi, s, ms).getTime();
}

const median = (xs: number[]): number => {
  const s = [...xs].sort((a, b) => a - b);
  return s.length ? s[Math.floor(s.length / 2)] : 0;
};

export interface Dies {
  rule: DieRule;
  cool(): Promise<Gate>;
  state(from: number, to: number, gate: Gate): DieState;
  close(): void;
}

export function watchDies(count: number, opts: { startC: number; stopC: number; intervalMs?: number }): Dies {
  const rule: DieRule = { dies: [...Array(count).keys()], startC: opts.startC, stopC: opts.stopC };
  const samples: Sample[] = [];
  const latest = new Map<number, Sample>();
  let failure = '';
  const query = 'timestamp,index,temperature.gpu,clocks.sm,power.draw,clocks_throttle_reasons.active';
  const smi: ChildProcess = spawn('nvidia-smi', [
    `--query-gpu=${query}`, '--format=csv,noheader,nounits', '-lms', String(opts.intervalMs ?? 100), '-i', rule.dies.join(','),
  ], { stdio: ['ignore', 'pipe', 'pipe'] });
  smi.on('error', (e) => { failure = `nvidia-smi: ${e.message}`; });
  smi.on('exit', (code) => { if (code && !failure) failure = `nvidia-smi exited with ${code}`; });
  createInterface({ input: smi.stdout! }).on('line', (line) => {
    const [stamp, die, c, mhz, w, bits] = line.split(',').map((s) => s.trim());
    const s: Sample = { t: epoch(stamp), die: Number(die), c: Number(c), mhz: Number(mhz), w: Number(w), bits: parseInt(bits, 16) };
    if (!Number.isFinite(s.t) || !Number.isFinite(s.c) || !Number.isFinite(s.mhz)) return;
    samples.push(s);
    latest.set(s.die, s);
  });

  const now = () => Object.fromEntries(rule.dies.map((d) => [d, latest.get(d)?.c ?? NaN]));
  const cooled = () => rule.dies.every((d) => (latest.get(d)?.c ?? Infinity) <= rule.startC);

  return {
    rule,

    async cool(): Promise<Gate> {
      const start = Date.now();
      while (!cooled()) {
        // No reading to wait on: state() says so, and the replies still count.
        if (failure) break;
        await new Promise((r) => setTimeout(r, 250));
      }
      return { waitedS: Math.round((Date.now() - start) / 1000), startC: now() };
    },

    state(from: number, to: number, gate: Gate): DieState {
      const reasons: string[] = [];
      const dies = rule.dies.map((die) => {
        let x = samples.filter((s) => s.die === die && s.t >= from && s.t <= to);
        // A window shorter than one pass over the dies: the last reading before it ends.
        if (!x.length) x = samples.filter((s) => s.die === die && s.t <= to).slice(-1);
        if (!x.length) {
          reasons.push(`no sample of GPU${die}${failure ? `: ${failure}` : ''}`);
          return { die, startC: gate.startC[die], medianMhz: 0, minMhz: 0, peakC: 0, meanW: 0, flags: [] };
        }
        const peakC = Math.max(...x.map((s) => s.c));
        const bits = x.reduce((b, s) => b | s.bits, 0);
        if (peakC >= rule.stopC) reasons.push(`GPU${die} reached ${peakC} °C`);
        if (bits & FAULT) reasons.push(`GPU${die} ${names(bits & FAULT).join('+')}`);
        const w = x.map((s) => s.w).filter(Number.isFinite);
        return {
          die,
          startC: gate.startC[die],
          medianMhz: median(x.map((s) => s.mhz)),
          minMhz: Math.min(...x.map((s) => s.mhz)),
          peakC,
          meanW: w.length ? Math.round(w.reduce((a, b) => a + b, 0) / w.length) : 0,
          flags: names(bits),
        };
      });
      return { valid: reasons.length === 0, reason: reasons.join('; '), cooledS: gate.waitedS, dies };
    },

    close(): void {
      smi.kill();
    },
  };
}
