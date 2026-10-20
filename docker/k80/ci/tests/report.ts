/**
 *   npx tsx report.ts [--input results/replies.json] [--output results/report.json]
 *                     [--test NAME]
 *   npx tsx report.ts --ready
 *
 *   --input   test.ts's --output file                (default results/replies.json)
 *   --output  a file path                            (default results/report.json)
 *   --test    a name                                 (default test)
 *   --ready   no value
 *
 *   GITHUB_SHA           a commit                    (default empty)
 *   GITHUB_RUN_ID        a workflow run's id         (default empty)
 *   GITHUB_STEP_SUMMARY  a file path, or unset
 */
import { appendFileSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { readable } from './lib/check.js';
import { dieNames } from './lib/dies.js';
import { closeJudge, judge, judgeReady } from './lib/judge.js';
import { arg, save, type Replies, type Result } from './lib/run.js';

if (process.argv.includes('--ready')) {
  try {
    await judgeReady();
    process.stderr.write('the judge is ready\n');
    process.exit(0);
  } catch (e) {
    process.stderr.write(`JUDGE NOT READY: ${e instanceof Error ? e.message : e}\n`);
    process.exit(1);
  }
}

const input = arg('input') ?? 'results/replies.json';
let run: Replies;
try {
  run = JSON.parse(readFileSync(input, 'utf-8'));
} catch {
  run = {
    test: arg('test') ?? 'test', about: '', commit: process.env.GITHUB_SHA ?? '', runId: process.env.GITHUB_RUN_ID ?? '', date: new Date().toISOString(),
    results: [], errors: [], bodyError: `no replies at ${input}: the test did not run`,
  };
}

function serverLine(): string | undefined {
  try {
    const s = JSON.parse(readFileSync(resolve(dirname(resolve(input)), 'server.json'), 'utf-8'));
    return `Server: ${s.server}, built from ${String(s.build).slice(0, 9)}${String(s.build).includes('modified') ? ' (modified)' : ''}; ` +
      `${s.model}, ${s.dtype}, ${s.dies} die(s), ${s.maxModelLen}-token context; ` +
      `weights ${s.weightsGib} GiB per die, KV cache ${s.kvTokens} tokens, ${s.vramMib} MiB VRAM on ${s.diesHolding} die(s).`;
  } catch {
    return undefined;
  }
}

// milliseconds, a whole number
if (run.results.some((r) => r.checks.content.pass)) await judgeReady(60_000).catch(() => {});
for (const r of run.results) {
  if (!r.checks.content.pass) {
    r.judge = { verdict: 'not-judged', reason: `not judged: ${r.checks.content.reason}` };
  } else {
    process.stderr.write(`  [judge] ${r.option} (${r.prompt})...\n`);
    r.judge = await judge(r.judgeTemplate, readable(r.reply));
  }
  r.pass = Object.values(r.checks).every((c) => c.pass) && r.judge.verdict === r.judgePass;
}
closeJudge();

const failed = run.results.filter((r) => !r.pass).length + run.errors.length;
const judged = run.results.filter((r) => r.judge.verdict === 'yes' || r.judge.verdict === 'no').length;
const abstained = run.results.filter((r) => r.judge.verdict === 'abstain').length;
const notJudged = run.results.filter((r) => r.judge.verdict === 'not-judged').length;
const empty = run.results.length === 0;
const invalid = run.results.filter((r) => r.dies && !r.dies.valid).length;
const server = serverLine();
const cell = (s: string) => s.replace(/\s+/g, ' ').replace(/\|/g, '\\|');

function diesCell(r: Result): string {
  if (!r.dies) return '—';
  const each = r.dies.dies.map((d) => {
    const flags = d.flags.filter((f) => f !== 'idle');
    return `GPU${d.die} ${d.medianMhz} MHz, ${d.startC}→${d.peakC} °C${flags.length ? `, ${flags.join('+')}` : ''}`;
  }).join('; ');
  return r.dies.valid ? each : `⚠ not comparable: ${r.dies.reason}; ${each}`;
}
const tps = (r: Result, v: unknown) => (r.dies && !r.dies.valid ? `${v} ⚠` : String(v));
const rule = run.dieRule;

const lines = [
  '',
  `## ${run.test}`,
  '',
  ...(run.about ? [run.about, ''] : []),
  `**${run.results.length} reply(s), ${judged} judged, ${abstained} abstained, ${notJudged} failed the check, ${failed} failed**` +
    (empty ? ' — no reply was recorded, so nothing was tested' : '') +
    (invalid ? `; **${invalid} with a die warning (⚠)**: their tok/s may show the die, not the server` : ''),
  '',
  '| Option | Model | Prompt | Checks | Judge | Pass | Prefill tok/s | Decode tok/s | Dies | In / out tok | More | Reply |',
  '|---|---|---|---|---|---|---|---|---|---|---|---|',
  ...run.results.map((r) => {
    const checks = Object.entries(r.checks).map(([k, v]) => `${k}: ${v.pass ? 'ok' : v.reason}`).join('; ');
    const { inTokens, outTokens, prefillTps, decodeTps, ...more } = r.metrics;
    const extra = Object.entries(more).map(([k, v]) => `${k}=${k === 'totalDecodeTps' ? tps(r, v) : v}`).join(' ');
    return `| ${r.option} | ${r.model} | ${r.prompt} | ${cell(checks)} | ${cell(r.judge.reason)} | ${r.pass ? 'PASS' : 'FAIL'} | ${tps(r, prefillTps)} | ${tps(r, decodeTps)} | ${cell(diesCell(r))} | ${inTokens} / ${outTokens} | ${extra} | ${cell(readable(r.reply)).slice(0, 80)} |`;
  }),
  ...run.errors.map((e) => `| ${e.option} | — | — | error: ${cell(e.error).slice(0, 120)} | — | FAIL | — | — | — | — | — | — |`),
  '',
  ...(rule
    ? [`Dies: each option ran on a server started once ${dieNames(rule.dies)} read ${rule.startC} °C or less with no model loaded (waiting as long as that took), or on the running one while they still read that; it shows each die's median SM clock, and its temperature at the start and at the peak, while it ran. ` +
        `A die at ${rule.stopC} °C, or a sw-power-cap, hw-slowdown, sw-thermal, hw-thermal or power-brake flag, marks the option's tok/s ⚠, not comparable with other runs. Read with nvidia-smi; nothing was set.`, '']
    : []),
  ...(server ? [server] : []),
  `Commit ${run.commit.slice(0, 9)}.`,
  ...(run.bodyError ? ['', `Run error: ${run.bodyError}`] : []),
  '',
];
const md = lines.join('\n');
process.stdout.write(md + '\n');
if (process.env.GITHUB_STEP_SUMMARY) appendFileSync(process.env.GITHUB_STEP_SUMMARY, md + '\n');

save(arg('output') ?? 'results/report.json', { ...run, judged, abstained, notJudged, failed, invalid });
process.exit(failed > 0 || empty || run.bodyError ? 1 : 0);
