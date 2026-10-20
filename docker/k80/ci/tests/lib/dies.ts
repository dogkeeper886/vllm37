import { spawn, type ChildProcess } from 'node:child_process';
import { createInterface } from 'node:readline';

const REASONS: [number, string][] = [
  [0x1, 'idle'], [0x2, 'app-clock'], [0x4, 'sw-power-cap'], [0x8, 'hw-slowdown'],
  [0x20, 'sw-thermal'], [0x40, 'hw-thermal'], [0x80, 'power-brake'],
];
// any of the REASONS bits, OR-ed together
const FAULT = 0x4 | 0x8 | 0x20 | 0x40 | 0x80;
const names = (bits: number): string[] => REASONS.filter(([b]) => bits & b).map(([, n]) => n);

export const dieNames = (dies: number[]): string =>
  dies.length > 1 ? `GPU${dies[0]}-${dies[dies.length - 1]}` : `GPU${dies[0]}`;

export interface DieRule {
  dies: number[];
  startC: number;
  stopC: number;
}

export interface Gate {
  waitedS: number;
  startC: Record<number, number>;
}

export interface DieState {
  valid: boolean;
  reason: string;
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
  ready(): Promise<void>;
  hot(): boolean;
  now(): Record<number, number>;
  state(from: number, to: number, gate: Gate): DieState;
  close(): void;
}

export function watchDies(count: number, opts: { startC: number; stopC: number; intervalMs?: number }): Dies {
  const rule: DieRule = { dies: [...Array(count).keys()], startC: opts.startC, stopC: opts.stopC };
  const samples: Sample[] = [];
  const latest = new Map<number, Sample>();
  let failure = '';
  const query = 'timestamp,index,temperature.gpu,clocks.sm,power.draw,clocks_throttle_reasons.active';
  // intervalMs: milliseconds, a whole number (default 100)
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

  return {
    rule,

    async ready(): Promise<void> {
      while (!failure && rule.dies.some((d) => !latest.has(d))) await new Promise((r) => setTimeout(r, 100));
    },

    hot(): boolean {
      return rule.dies.some((d) => (latest.get(d)?.c ?? -Infinity) > rule.startC);
    },

    now(): Record<number, number> {
      return Object.fromEntries(rule.dies.map((d) => [d, latest.get(d)?.c ?? NaN]));
    },

    state(from: number, to: number, gate: Gate): DieState {
      const reasons: string[] = [];
      const dies = rule.dies.map((die) => {
        let x = samples.filter((s) => s.die === die && s.t >= from && s.t <= to);
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
