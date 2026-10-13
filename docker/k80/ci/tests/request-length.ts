/**
 * request length: how fast are a short prompt and a long one?
 *
 *   npx tsx request-length.ts [--output results/request-length.json]
 *
 * One server on the first row of the use-case table, one call per option.
 * Options are the two throughput prompts of ollama37: speech-rewrite (~870
 * tokens in) and farewell-summary (~8k tokens in), 100 tokens out each.
 */
import { forEachOption, runTest } from './lib/run.js';
import { generate, modelTest, serve } from './lib/vllm.js';

await runTest('request length', 'A short prompt, then a long one: prefill and decode speed of each.', async () => {
  const { maxModelLen } = await serve();
  await modelTest('start');
  await forEachOption(['speech-rewrite', 'farewell-summary'], async (option) => {
    const [r] = await generate(option, option);
    r.checks.window = { pass: r.metrics.inTokens + r.metrics.outTokens <= maxModelLen, reason: `${r.metrics.inTokens} of ${maxModelLen} tokens` };
  });
});
