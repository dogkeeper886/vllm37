/**
 * The test: under the condition its inputs set, does vllm37 still give a real
 * answer and not garbage?
 *
 *   npx tsx test.ts [--test NAME] [--prompt a,b] [--calls 1,2] [--stagger 0,20]
 *                   [--dies 2] [--start-temp 50] [--stop-temp 80]
 *                   [--server docker|local] [--output results/replies.json]
 *
 * Talks to a server that is already up (VLLM37_HOST, default localhost:8000);
 * docker/k80/ci/service-start.sh starts it. Inputs, each a value or a list:
 *
 *   --prompt   prompt name in prompts.yaml            (default speech-rewrite)
 *   --calls    how many calls send the prompt at once (default 1)
 *   --stagger  tokens a call has streamed before the next is sent; 0 sends the
 *              calls together                         (default 0)
 *
 * and the dies, one value each (lib/dies.ts):
 *
 *   --dies        how many dies the server takes, GPU0 first        (default 1)
 *   --start-temp  whole °C from 1, below --stop-temp                (default 60)
 *   --stop-temp   °C at which a die makes the option invalid         (default 80)
 *   --server      docker or local                                   (default docker)
 *
 * Sends short-answer first, the one reply with a finished answer, then the timed
 * calls: one option per combination of the listed values. Every reply is checked
 * here and judged by report.ts once the GPUs are back.
 */
import { spawn } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { dieNames, watchDies } from './lib/dies.js';
import { promptNames } from './lib/prompts.js';
import { arg, forEachOption, runTest, type Result } from './lib/run.js';
import { generate, served, type GenerateOptions } from './lib/vllm.js';

const list = (name: string, fallback: string): string[] =>
  (arg(name) || fallback).split(',').map((s) => s.trim()).filter(Boolean);
const counts = (name: string, fallback: string, min: number): number[] =>
  list(name, fallback).map((v) => {
    if (!/^[0-9]+$/.test(v) || Number(v) < min) throw new Error(`--${name} takes whole numbers from ${min}, got "${v}"`);
    return Number(v);
  });

const prompts = list('prompt', 'speech-rewrite');
const unknown = prompts.filter((p) => !promptNames().includes(p));
if (unknown.length) throw new Error(`--prompt must be one of ${promptNames().join(', ')}, got "${unknown.join(', ')}"`);
const calls = counts('calls', '1', 1);
const stagger = counts('stagger', '0', 0);
const one = (name: string, fallback: string, min: number): number => {
  const v = counts(name, fallback, min);
  if (v.length !== 1) throw new Error(`--${name} takes one value, got "${v.join(',')}"`);
  return v[0];
};
const dies = watchDies(one('dies', '1', 1), { startC: one('start-temp', '60', 1), stopC: one('stop-temp', '80', 1) });
const { startC, stopC } = dies.rule;
if (stopC <= startC) throw new Error(`--stop-temp (${stopC}) must be above --start-temp (${startC})`);
const server = arg('server') ?? 'docker';
if (server !== 'docker' && server !== 'local') throw new Error(`--server must be docker or local, got "${server}"`);

// One option per combination, named by the inputs that hold more than one value.
const options = prompts.flatMap((prompt) => calls.flatMap((n) => stagger.map((after) => ({ prompt, calls: n, stagger: after }))));
const label = (o: (typeof options)[number]): string =>
  [
    prompts.length > 1 ? o.prompt : '',
    calls.length > 1 ? `calls=${o.calls}` : '',
    stagger.length > 1 ? `stagger=${o.stagger}` : '',
  ].filter(Boolean).join(', ') || o.prompt;

const about = `Inputs: prompt=${prompts.join(',')}; calls=${calls.join(',')}; stagger=${stagger.join(',')}. ` +
  `Each option runs on a server started once ${dieNames(dies.rule.dies)} read ${startC} °C or less with no model loaded, or on the running one while they still read that; ` +
  `a die at ${stopC} °C or a fault flag marks its tok/s ⚠, not comparable.`;

const CI = fileURLToPath(new URL('..', import.meta.url));
const RESULTS = process.env.VLLM37_RESULTS ?? join(CI, 'tests', 'results');
const cooledS = (): number => {
  const file = join(RESULTS, 'cooled.json');
  return existsSync(file) ? JSON.parse(readFileSync(file, 'utf-8')).waitedS : 0;
};

async function restart(): Promise<void> {
  const env = { ...process.env, GITHUB_STEP_SUMMARY: '', GITHUB_ENV: '' };
  for (const [script, args] of [['service-stop.sh', ['--server']], ['service-start.sh', [server]]] as const) {
    const code = await new Promise<number | null>((done) => {
      spawn(join(CI, script), [...args], { env, stdio: ['ignore', 2, 2] }).on('exit', done);
    });
    if (code !== 0) throw new Error(`${script} exited with ${code}`);
  }
}

let fresh = true;

async function timed(option: string, name: string, opts?: GenerateOptions): Promise<Result[]> {
  await dies.ready();
  let waitedS = 0;
  if (fresh) {
    waitedS = cooledS();
  } else if (dies.hot()) {
    process.stderr.write(`  a die reads over ${startC} °C: restarting the server so the dies cool with no model loaded\n`);
    await restart();
    waitedS = cooledS();
  }
  fresh = false;
  const gate = { waitedS, startC: dies.now() };
  process.stderr.write(`  cooled ${gate.waitedS} s; dies: ${Object.entries(gate.startC).map(([d, c]) => `GPU${d} ${c} °C`).join(', ')}\n`);
  const from = Date.now();
  const results = await generate(option, name, opts);
  const state = dies.state(from, Date.now(), gate);
  if (!state.valid) process.stderr.write(`  INVALID die state: ${state.reason}\n`);
  for (const r of results) r.dies = state;
  return results;
}

await runTest(arg('test') ?? 'test', about, async () => {
  try {
    const { maxModelLen } = await served();
    // A prompt that did not fit the context was cut, so the reply answers a prompt nobody sent.
    const fits = (r: Result): void => {
      if (maxModelLen) r.checks.window = { pass: r.metrics.inTokens + r.metrics.outTokens <= maxModelLen, reason: `${r.metrics.inTokens} of ${maxModelLen} tokens` };
    };
    (await timed('model test', 'short-answer')).forEach(fits);
    await forEachOption(options.map(label), async (name, i) => {
      const o = options[i];
      (await timed(name, o.prompt, { calls: o.calls, stagger: o.stagger })).forEach(fits);
    });
  } finally {
    dies.close();
  }
}, dies.rule);
