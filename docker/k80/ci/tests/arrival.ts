/**
 * arrival: does a call that arrives while another is decoding join the running batch?
 *
 *   npx tsx arrival.ts [--output results/arrival.json]
 *
 * One server on the first row of the use-case table, two calls per option with
 * the speech-rewrite prompt. Options: together (both sent at once), then
 * staggered (the second sent once the first has streamed some tokens).
 */
import { forEachOption, runTest } from './lib/run.js';
import { generate, modelTest, serve } from './lib/vllm.js';

await runTest('arrival', 'Two calls sent together, then the second sent while the first is decoding.', async () => {
  await serve();
  await modelTest('start');
  await forEachOption(['together', 'staggered'] as const, (option) => generate(option, 'speech-rewrite', { calls: 2, arrival: option }));
});
