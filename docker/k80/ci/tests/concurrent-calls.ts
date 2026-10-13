/**
 * concurrent calls: does a second call at the same time raise the total decode rate?
 *
 *   npx tsx concurrent-calls.ts [--output results/concurrent-calls.json]
 *
 * One server on the first row of the use-case table. Options: 1 call, then 2
 * calls sent together, each with the speech-rewrite prompt. Every reply is
 * checked and judged by the run.
 */
import { forEachOption, runTest } from './lib/run.js';
import { generate, modelTest, serve } from './lib/vllm.js';

await runTest('concurrent calls', 'One call, then two calls sent together: the total decode rate of each.', async () => {
  await serve();
  await modelTest('start');
  await forEachOption(['1 call', '2 calls'], (option, i) => generate(option, 'speech-rewrite', { calls: i + 1 }));
});
