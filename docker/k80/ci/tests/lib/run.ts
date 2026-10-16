/**
 * A test run, in two halves with the service step between them.
 * From ollama37's cicd/tests/lib/run.ts; a result here belongs to an option of
 * the test, not to a model, and the run no longer starts or stops anything.
 *
 *   test.ts    runTest('die-count', about, body)
 *     ├─ body: generate(option, name) → reply → check → recorded
 *     └─ replies written to --output; exit 1 on an error, a failed check or no reply
 *
 *   service-stop.sh gives the GPUs back
 *
 *   report.ts
 *     ├─ judge every recorded reply
 *     └─ table + JSON + exit       "N replies, N judged, K failed"
 *
 * This file is the test half. It starts no judge and no server. A test cannot
 * skip the judge: generate() records into the run, runTest writes every record,
 * and report.ts judges every record before it reports.
 */
import { spawnSync } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import type { Reply, Verdict } from './check.js';
import type { DieRule, DieState } from './dies.js';
import type { Judgment } from './judge.js';

export interface Result {
  /** The option of the test this reply belongs to, e.g. "calls=2". */
  option: string;
  model: string;
  prompt: string;
  judgeName: string;
  /** The judge's answer that passes the reply (prompts.yaml). */
  judgePass: 'yes' | 'no';
  /** The question the judge is asked about this reply. */
  judgeTemplate: string;
  reply: Reply;
  /** `content` is the script check generate() ran; a test may add its own (never remove). */
  checks: Record<string, Verdict>;
  metrics: { inTokens: number; outTokens: number; prefillTps: number; decodeTps: number } & Record<string, unknown>;
  /** `not-judged`: the script check already failed it, so it was never sent. Never passes. */
  judge: Judgment | { verdict: 'pending' | 'not-judged'; reason: string };
  pass: boolean;
  /** The dies while the option ran; an invalid one's tok/s says nothing about the server. */
  dies?: DieState;
}

type Recorded = Omit<Result, 'checks' | 'judge' | 'pass'> & { check: Verdict };

/** What test.ts writes and report.ts reads. */
export interface Replies {
  test: string;
  about: string;
  commit: string;
  runId: string;
  date: string;
  results: Result[];
  errors: { option: string; error: string }[];
  bodyError?: string;
  /** The temperatures each option was held to (lib/dies.ts). */
  dieRule?: DieRule;
}

let current: Pick<Replies, 'test' | 'results' | 'errors'> | undefined;

/** Called by generate() before it asks the model: outside a run no reply would reach the judge. */
export function assertInRun(): NonNullable<typeof current> {
  if (!current) throw new Error('generate() only runs inside runTest(): outside it no reply would be judged');
  return current;
}

/** Called by generate() with the reply. */
export function record(r: Recorded): Result {
  const run = assertInRun();
  const { check, ...rest } = r;
  const result: Result = { ...rest, checks: { content: check }, judge: { verdict: 'pending', reason: 'not judged yet' }, pass: false };
  run.results.push(result);
  return result;
}

/** Run `each` for every option, recording an option that throws as a failure rather than ending the run. */
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

/** The value after --name on the command line. */
export function arg(name: string): string | undefined {
  const i = process.argv.indexOf(`--${name}`);
  return i > 0 ? process.argv[i + 1] : undefined;
}

export function save(path: string, value: unknown): void {
  mkdirSync(dirname(resolve(path)), { recursive: true });
  writeFileSync(path, JSON.stringify(value, null, 2));
}

/**
 * The test half: run `body`, which talks to a server that is already up, and
 * write every reply it recorded. `about` is the one line under the title.
 */
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

  const commit = process.env.GITHUB_SHA ?? spawnSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf-8' }).stdout?.trim() ?? '';
  const replies: Replies = {
    test, about, commit, runId: process.env.GITHUB_RUN_ID ?? '', date: new Date().toISOString(),
    results: run.results, errors: run.errors, bodyError, dieRule,
  };
  save(arg('output') ?? 'results/replies.json', replies);

  const unchecked = run.results.filter((r) => !Object.values(r.checks).every((c) => c.pass)).length;
  process.stderr.write(`${run.results.length} reply(s) recorded, ${unchecked} failed a check, ${run.errors.length} option(s) errored\n`);
  process.exit(bodyError || run.errors.length || unchecked || run.results.length === 0 ? 1 : 0);
}
