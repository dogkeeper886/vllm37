/**
 * A test run, start to finish.
 * From ollama37's cicd/tests/lib/run.ts; a result here belongs to an option of
 * the test (one cell of a column in the use-case table), not to a model.
 *
 *   runTest('concurrent calls', about, body)
 *     ├─ judge reachable?    before the GPU is taken
 *     ├─ yield GPU           docker/k80/ci/gpu-yield.sh
 *     ├─ body: serve → generate(option, name) → reply → check → recorded
 *     ├─ restore GPU         docker/k80/ci/gpu-restore.sh
 *     ├─ judge every recorded reply
 *     └─ report + exit       "N replies, N judged, K failed"
 *
 * The judge runs after the GPU is back because on a shared host its server is the
 * one the test borrowed the card from. A test cannot skip it: generate() records
 * into the run, and runTest judges every record before it reports. The run fails
 * on a failed check, a "yes" to the garbage question, an abstain, an option that
 * errored, or a run that recorded no reply at all -- "0 failed" means every reply
 * was checked and judged.
 */
import { spawnSync } from 'node:child_process';
import { appendFileSync, mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { readable, type Reply, type Verdict } from './check.js';
import { judge, judgeReady, closeJudge, type Judgment } from './judge.js';

export interface Result {
  /** The option of the test this reply belongs to, e.g. "2 calls". */
  option: string;
  model: string;
  prompt: string;
  judgeName: string;
  /** The judge's answer that passes the reply (prompts.yaml). */
  judgePass: 'yes' | 'no';
  reply: Reply;
  /** `content` is the script check generate() ran; a test may add its own (never remove). */
  checks: Record<string, Verdict>;
  metrics: { inTokens: number; outTokens: number; prefillTps: number; decodeTps: number } & Record<string, unknown>;
  /** `not-judged`: the script check already failed it, so it was never sent. Never passes. */
  judge: Judgment | { verdict: 'pending' | 'not-judged'; reason: string };
  pass: boolean;
}

type Recorded = Omit<Result, 'checks' | 'judge' | 'pass'> & { judgeTemplate: string; check: Verdict };

interface Run { test: string; results: Result[]; templates: Map<Result, string>; errors: { option: string; error: string }[]; fixed: string[] }
let current: Run | undefined;

/** Called by generate() before it asks the model: outside a run no one would judge the reply. */
export function assertInRun(): Run {
  if (!current) throw new Error('generate() only runs inside runTest(): outside it no reply would be judged');
  return current;
}

/** Called by generate() with the reply. */
export function record(r: Recorded): Result {
  const run = assertInRun();
  const { judgeTemplate, check, ...rest } = r;
  const result: Result = { ...rest, checks: { content: check }, judge: { verdict: 'pending', reason: 'not judged yet' }, pass: false };
  run.results.push(result);
  run.templates.set(result, judgeTemplate);
  return result;
}

/** Work to do just before the GPU is given back, while the server under test still exists. */
const beforeRestoreHooks: (() => void)[] = [];
export function beforeRestore(hook: () => void): void {
  beforeRestoreHooks.push(hook);
}

/** Called by serve() with what the server was started with: the line under the table. */
export function heldFixed(settings: string): void {
  const run = assertInRun();
  if (!run.fixed.includes(settings)) run.fixed.push(settings);
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

/** docker/k80/ci, where the GPU hand-over scripts live. Their output goes to stderr: stdout is the report. */
const SCRIPTS = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
function shell(script: string): void {
  const r = spawnSync('bash', [resolve(SCRIPTS, script)], { stdio: ['ignore', 2, 2] });
  if (r.status !== 0) throw new Error(`${script} exited ${r.status}`);
}

/** The value after --name on the command line. */
export function arg(name: string): string | undefined {
  const i = process.argv.indexOf(`--${name}`);
  return i > 0 ? process.argv[i + 1] : undefined;
}

/** `about` is the one line under the title that says what the test asks. */
export async function runTest(test: string, about: string, body: () => Promise<void>): Promise<never> {
  current = { test, results: [], templates: new Map(), errors: [], fixed: [] };
  const run = current;
  let bodyError: string | undefined;
  let restoreError: string | undefined;

  // The yield is inside the try: it stops the host's own servers before the test
  // starts the one under test, so a yield that fails halfway must still be restored.
  try {
    await judgeReady();
    shell('gpu-yield.sh');
    await body();
  } catch (e) {
    bodyError = e instanceof Error ? e.message : String(e);
    process.stderr.write(`RUN ERROR: ${bodyError}\n`);
  } finally {
    // A failed restore must not throw away a run's replies: record it, judge and
    // report as usual, and fail the run. The workflows restore again in an
    // always() step, which also covers a job that was cancelled or timed out.
    try {
      for (const hook of beforeRestoreHooks) hook();
      shell('gpu-restore.sh');
    } catch (e) {
      restoreError = e instanceof Error ? e.message : String(e);
      process.stderr.write(`RESTORE ERROR: ${restoreError}\n`);
    }
  }

  // Judge every reply the run recorded. A reply the script check failed is not
  // sent -- it fails anyway, and a judge asked to quote a loop loops with it.
  // The judge's server may be the one the restore just started: give it a minute.
  if (run.results.some((r) => r.checks.content.pass)) await judgeReady(60_000).catch(() => {});
  for (const r of run.results) {
    if (!r.checks.content.pass) {
      r.judge = { verdict: 'not-judged', reason: `not judged: ${r.checks.content.reason}` };
    } else {
      process.stderr.write(`  [judge] ${r.option} (${r.prompt})...\n`);
      r.judge = await judge(run.templates.get(r)!, readable(r.reply));
    }
    r.pass = Object.values(r.checks).every((c) => c.pass) && r.judge.verdict === r.judgePass;
  }
  closeJudge();

  const failed = run.results.filter((r) => !r.pass).length + run.errors.length;
  const judged = run.results.filter((r) => r.judge.verdict === 'yes' || r.judge.verdict === 'no').length;
  const abstained = run.results.filter((r) => r.judge.verdict === 'abstain').length;
  const notJudged = run.results.filter((r) => r.judge.verdict === 'not-judged').length;
  const empty = run.results.length === 0;
  const commit = process.env.GITHUB_SHA ?? spawnSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf-8' }).stdout?.trim() ?? '';
  const cell = (s: string) => s.replace(/\s+/g, ' ').replace(/\|/g, '\\|');

  const lines = [
    `## ${test}`,
    '',
    about,
    '',
    `**${run.results.length} reply(s), ${judged} judged, ${abstained} abstained, ${notJudged} failed the check, ${failed} failed**` +
      (empty ? ' — no reply was recorded, so nothing was tested' : ''),
    '',
    '| Option | Model | Prompt | Checks | Judge | Pass | Prefill tok/s | Decode tok/s | In / out tok | More | Reply |',
    '|---|---|---|---|---|---|---|---|---|---|---|',
    ...run.results.map((r) => {
      const checks = Object.entries(r.checks).map(([k, v]) => `${k}: ${v.pass ? 'ok' : v.reason}`).join('; ');
      const { inTokens, outTokens, prefillTps, decodeTps, ...more } = r.metrics;
      const extra = Object.entries(more).map(([k, v]) => `${k}=${v}`).join(' ');
      return `| ${r.option} | ${r.model} | ${r.prompt} | ${cell(checks)} | ${cell(r.judge.reason)} | ${r.pass ? 'PASS' : 'FAIL'} | ${prefillTps} | ${decodeTps} | ${inTokens} / ${outTokens} | ${extra} | ${cell(readable(r.reply)).slice(0, 80)} |`;
    }),
    ...run.errors.map((e) => `| ${e.option} | — | — | error: ${cell(e.error).slice(0, 120)} | — | FAIL | — | — | — | — | — |`),
    ...(run.fixed.length ? ['', `Server started with: ${run.fixed.join('; then ')}. Commit ${commit.slice(0, 9)}.`] : []),
    ...(bodyError ? ['', `Run error: ${bodyError}`] : []),
    ...(restoreError ? ['', `GPU restore failed: ${restoreError}`] : []),
    '',
  ];
  const md = lines.join('\n');
  process.stdout.write(md + '\n');
  if (process.env.GITHUB_STEP_SUMMARY) appendFileSync(process.env.GITHUB_STEP_SUMMARY, md + '\n');

  const out = arg('output');
  if (out) {
    mkdirSync(dirname(resolve(out)), { recursive: true });
    writeFileSync(out, JSON.stringify({
      test, about, commit, runId: process.env.GITHUB_RUN_ID ?? '', date: new Date().toISOString(),
      servers: run.fixed, judged, abstained, notJudged, failed, results: run.results, errors: run.errors, bodyError, restoreError,
    }, null, 2));
  }
  process.exit(failed > 0 || empty || bodyError || restoreError ? 1 : 0);
}
