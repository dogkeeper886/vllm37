/**
 * The summary of a test-all.yml or test-use-case.yml run: one table over the
 * reports of every test job.
 *
 *   npx tsx summary.ts [--input reports]
 *
 * Reads <input>/<artifact>/report.json and the server.json beside it, as the
 * test-end action uploads them, one row per option of each test. NEEDS, the
 * workflow's needs context, names the jobs that left no report. It judges
 * nothing and fails nothing: each test job already passed or failed on its
 * judged replies. An option whose dies ran hot or throttled has its tok/s
 * marked ⚠, not comparable with other runs.
 */
import { appendFileSync, existsSync, readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { arg, type Replies } from './lib/run.js';

type Report = Replies & { judged: number; failed: number };
interface Server { server: string; build: string; model: string; dtype: string; dies: number; maxModelLen: number }

const dir = arg('input') ?? 'reports';
const read = <T>(path: string): T | undefined => {
  try {
    return JSON.parse(readFileSync(path, 'utf-8')) as T;
  } catch {
    return undefined;
  }
};

const runs = (existsSync(dir) ? readdirSync(dir) : [])
  .map((name) => ({ report: read<Report>(join(dir, name, 'report.json')), server: read<Server>(join(dir, name, 'server.json')) }))
  .filter((r): r is { report: Report; server: Server | undefined } => r.report !== undefined)
  .sort((a, b) => a.report.date.localeCompare(b.report.date));

const cell = (s: string) => s.replace(/\s+/g, ' ').replace(/\|/g, '\\|');
const build = (s?: Server) => (s ? `${s.server} \`${s.build.slice(0, 9)}\`${s.build.includes('modified') ? ' modified' : ''}` : '—');
const served = (s?: Server) => (s ? `${s.dtype}, ${s.dies} die(s)` : 'no server');

const rows: string[] = [];
for (const { report, server } of runs) {
  const options = [...new Set([...report.results.map((r) => r.option), ...report.errors.map((e) => e.option)])];
  for (const option of options) {
    const results = report.results.filter((r) => r.option === option);
    const errors = report.errors.filter((e) => e.option === option);
    const judged = results.filter((r) => r.judge.verdict === 'yes' || r.judge.verdict === 'no').length;
    const failed = results.filter((r) => !r.pass).length + errors.length;
    const m = results[0]?.metrics;
    const d = results[0]?.dies;
    const total = m ? String(m.totalDecodeTps ?? m.decodeTps) : '—';
    const decode = m && d && !d.valid ? `${total} ⚠` : total;
    const dies = d
      ? `${d.valid ? '' : `⚠ not comparable: ${d.reason}; `}peak ${Math.max(...d.dies.map((x) => x.peakC))} °C, ` +
        `clock ≥ ${Math.min(...d.dies.map((x) => x.medianMhz))} MHz`
      : '—';
    rows.push(`| ${report.test} | ${cell(option)}; ${served(server)} | ${results.length} | ${judged} | ${failed} | ${decode} | ${cell(dies)} | ${build(server)} |`);
  }
  if (!options.length) {
    rows.push(`| ${report.test} | ${cell(report.bodyError ?? 'no reply')}; ${served(server)} | 0 | 0 | 1 | — | — | ${build(server)} |`);
  }
}

const models = [...new Set(runs.flatMap(({ server: s }) => (s ? [`${s.model}, ${s.maxModelLen}-token context`] : [])))];
const needs = JSON.parse(process.env.NEEDS ?? '{}') as Record<string, { result: string }>;
const jobs = Object.entries(needs).filter(([name]) => name !== 'plan').map(([name, j]) => `${name}: ${j.result}`);

const md = [
  '',
  '## All tests',
  '',
  ...(models.length ? [`Model: ${models.join('; ')}.`, ''] : []),
  '| Job | Condition | Replies | Judged | Failed | Decode tok/s | Dies | Build |',
  '|---|---|---|---|---|---|---|---|',
  ...(rows.length ? rows : ['| — | no report was uploaded | — | — | — | — | — | — |']),
  '',
  'Decode tok/s is the total of all calls in the option. Failed counts the replies that failed a check or the judge, and the errors.',
  'Dies: the hottest die\'s peak temperature and the lowest median SM clock while the option ran. A ⚠ tok/s ran with a die that was hot, slowed or not sampled (reason in Dies), so it is not comparable with other runs.',
  ...(jobs.length ? ['', `Jobs: ${jobs.join(' · ')}.`] : []),
  '',
].join('\n');
process.stdout.write(md + '\n');
if (process.env.GITHUB_STEP_SUMMARY) appendFileSync(process.env.GITHUB_STEP_SUMMARY, md + '\n');
