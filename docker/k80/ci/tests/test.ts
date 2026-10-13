/**
 * The test: under the condition its inputs set, does vllm37 still give a real
 * answer and not garbage?
 *
 *   npx tsx test.ts [--test NAME] [--prompt a,b] [--calls 1,2] [--stagger 0,20]
 *                   [--output results/replies.json]
 *
 * Talks to a server that is already up (VLLM37_HOST, default localhost:8000);
 * docker/k80/ci/service-start.sh starts it. Inputs, each a value or a list:
 *
 *   --prompt   prompt name in prompts.yaml            (default speech-rewrite)
 *   --calls    how many calls send the prompt at once (default 1)
 *   --stagger  tokens a call has streamed before the next is sent; 0 sends the
 *              calls together                         (default 0)
 *
 * Sends short-answer first, the one reply with a finished answer, then the timed
 * calls: one option per combination of the listed values. Every reply is checked
 * here and judged by report.ts once the GPUs are back.
 */
import { promptNames } from './lib/prompts.js';
import { arg, forEachOption, runTest, type Result } from './lib/run.js';
import { generate, served } from './lib/vllm.js';

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

// One option per combination, named by the inputs that hold more than one value.
const options = prompts.flatMap((prompt) => calls.flatMap((n) => stagger.map((after) => ({ prompt, calls: n, stagger: after }))));
const label = (o: (typeof options)[number]): string =>
  [
    prompts.length > 1 ? o.prompt : '',
    calls.length > 1 ? `calls=${o.calls}` : '',
    stagger.length > 1 ? `stagger=${o.stagger}` : '',
  ].filter(Boolean).join(', ') || o.prompt;

const about = `Inputs: prompt=${prompts.join(',')}; calls=${calls.join(',')}; stagger=${stagger.join(',')}.`;

await runTest(arg('test') ?? 'test', about, async () => {
  const { maxModelLen } = await served();
  // A prompt that did not fit the context was cut, so the reply answers a prompt nobody sent.
  const fits = (r: Result): void => {
    if (maxModelLen) r.checks.window = { pass: r.metrics.inTokens + r.metrics.outTokens <= maxModelLen, reason: `${r.metrics.inTokens} of ${maxModelLen} tokens` };
  };
  (await generate('model test', 'short-answer')).forEach(fits);
  await forEachOption(options.map(label), async (name, i) => {
    const o = options[i];
    (await generate(name, o.prompt, { calls: o.calls, stagger: o.stagger })).forEach(fits);
  });
});
