import { spawnSync } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import type { Reply, Verdict } from './check.js';
import type { DieRule, DieState } from './dies.js';
import type { Judgment } from './judge.js';

export interface Result {
  option: string;
  model: string;
  prompt: string;
  judgeName: string;
  judgePass: 'yes' | 'no';
  judgeTemplate: string;
  reply: Reply;
  checks: Record<string, Verdict>;
  metrics: { inTokens: number; outTokens: number; prefillTps: number; decodeTps: number } & Record<string, unknown>;
  judge: Judgment | { verdict: 'pending' | 'not-judged'; reason: string };
  pass: boolean;
  dies?: DieState;
}

type Recorded = Omit<Result, 'checks' | 'judge' | 'pass'> & { check: Verdict };

export interface Replies {
  test: string;
  about: string;
  commit: string;
  runId: string;
  date: string;
  results: Result[];
  errors: { option: string; error: string }[];
  bodyError?: string;
  dieRule?: DieRule;
}

let current: Pick<Replies, 'test' | 'results' | 'errors'> | undefined;

export function assertInRun(): NonNullable<typeof current> {
  if (!current) throw new Error('generate() only runs inside runTest(): outside it no reply would be judged');
  return current;
}

export function record(r: Recorded): Result {
  const run = assertInRun();
  const { check, ...rest } = r;
  const result: Result = { ...rest, checks: { content: check }, judge: { verdict: 'pending', reason: 'not judged yet' }, pass: false };
  run.results.push(result);
  return result;
}

export async function forEachOption<T extends string>(options: readonly T[], each: (option: T, index: number) => Promise<unknown>): Promise<void> {
  for (const [i, option] of options.entries()) {
    process.stderr.write(`--- ${option} ---\n`);
    try {
      await each(option, i);
    } catch (e) {
      const error = e instanceof Error ? e.message : String(e);
      process.stderr.write(`  ERROR: ${error}\n`);
      current!.errors.push({ option, error });
    }
  }
}

export function arg(name: string): string | undefined {
  const i = process.argv.indexOf(`--${name}`);
  return i > 0 ? process.argv[i + 1] : undefined;
}

export function save(path: string, value: unknown): void {
  mkdirSync(dirname(resolve(path)), { recursive: true });
  writeFileSync(path, JSON.stringify(value, null, 2));
}

export async function runTest(test: string, about: string, body: () => Promise<void>, dieRule?: DieRule): Promise<never> {
  current = { test, results: [], errors: [] };
  const run = current;
  let bodyError: string | undefined;
  try {
    await body();
  } catch (e) {
    bodyError = e instanceof Error ? e.message : String(e);
    process.stderr.write(`RUN ERROR: ${bodyError}\n`);
  }

  // GITHUB_SHA: a commit (default git rev-parse HEAD)
  const commit = process.env.GITHUB_SHA ?? spawnSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf-8' }).stdout?.trim() ?? '';
  const replies: Replies = {
    // GITHUB_RUN_ID: a workflow run's id (default empty)
    test, about, commit, runId: process.env.GITHUB_RUN_ID ?? '', date: new Date().toISOString(),
    results: run.results, errors: run.errors, bodyError, dieRule,
  };
  save(arg('output') ?? 'results/replies.json', replies);

  const unchecked = run.results.filter((r) => !Object.values(r.checks).every((c) => c.pass)).length;
  process.stderr.write(`${run.results.length} reply(s) recorded, ${unchecked} failed a check, ${run.errors.length} option(s) errored\n`);
  process.exit(bodyError || run.errors.length || unchecked || run.results.length === 0 ? 1 : 0);
}
