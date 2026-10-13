/**
 * weight dtype: does the model serve with FP16-stored weights as it does with FP32?
 *
 *   npx tsx weight-dtype.ts [--output results/weight-dtype.json]
 *
 * Options: float32, then half (weights and KV cache in 2 bytes per value, math
 * in FP32). Each starts its own server with the same model, runs the model test
 * and one speech-rewrite call.
 */
import { forEachOption, runTest } from './lib/run.js';
import { generate, modelTest, serve } from './lib/vllm.js';

await runTest('weight dtype', 'FP32 weights, then FP16-stored weights: memory held and speed of each.', () =>
  forEachOption(['float32', 'half'] as const, async (option) => {
    await serve({ dtype: option });
    await modelTest(option);
    await generate(option, 'speech-rewrite');
  }),
);
