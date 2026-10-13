/**
 * die count: does the model serve from two dies of one board as it does from one?
 *
 *   npx tsx die-count.ts [--output results/die-count.json]
 *
 * Options: 1 die (tensor parallel size 1), then 2 dies (size 2, both dies of one
 * board). Each starts its own server, runs the model test and one speech-rewrite
 * call. Never more than 2: four dies run both boards from one power cable and
 * have halted this machine.
 */
import { serverVram } from './lib/gpu.js';
import { forEachOption, runTest } from './lib/run.js';
import { generate, modelTest, serve } from './lib/vllm.js';

await runTest('die count', 'One die, then both dies of one board: memory held and speed of each.', () =>
  forEachOption(['1 die', '2 dies'], async (option, i) => {
    const dies = i === 0 ? 1 : 2;
    await serve({ dies });
    const r = await modelTest(option);
    const vram = serverVram();
    r.checks.dies = { pass: vram.dies === dies, reason: `${vram.totalMib} MiB across ${vram.dies} die(s), asked for ${dies}` };
    await generate(option, 'speech-rewrite');
  }),
);
