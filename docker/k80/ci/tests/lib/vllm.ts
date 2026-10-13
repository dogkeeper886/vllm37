/**
 * The only code that talks to the server under test.
 * Takes the place of ollama37's cicd/tests/lib/ollama.ts.
 *
 *   serve({ dtype: 'half' })
 *     └─ docker/k80/container-up.sh with the first row of the use-case table,
 *        changed by what the option names
 *
 *   generate('2 calls', 'speech-rewrite', { calls: 2 })
 *     └─ prompts.yaml['speech-rewrite'] → text, options, judge question
 *          └─ /v1/chat/completions, streamed → reply, prefill and decode speed
 *               └─ check(reply)
 *                    └─ recorded for the judge, who runs after the GPU is back
 *                         └─ Result { reply, check, judge, pass }
 *
 * `generate` and `modelTest` are the only exports that return model text, and
 * they only run inside runTest, so every reply is checked and every reply is
 * judged. `serve` and `serverInfo` produce no model text.
 */
import { spawnSync } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import http from 'node:http';
import { basename, dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { check, type Reply } from './check.js';
import { serverVram } from './gpu.js';
import { prompt } from './prompts.js';
import { arg, assertInRun, beforeRestore, heldFixed, record, type Result } from './run.js';

export const HOST = process.env.VLLM37_HOST ?? 'http://localhost:8000';
const CONTAINER = process.env.VLLM37_CONTAINER || 'vllm37';
/** docker/k80, where container-up.sh and container-down.sh live. */
const K80 = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..');
/** Tokens per KV cache block (vLLM's default --block-size). */
const BLOCK_TOKENS = 16;
/** A staggered call is sent once the call before it has streamed this many tokens. */
const STAGGER_AFTER = 20;

/** What the server is started with. */
export interface Settings {
  model: string;
  dtype: 'float32' | 'half';
  /** Tensor parallel size: 1, or 2 for both dies of one board. Never more. */
  dies: 1 | 2;
  /** Context length. One value for every test, so only the option under test differs. */
  maxModelLen: number;
}

/** The first row of the use-case table. A test changes one field of it per option. */
export const FIRST_ROW: Settings = { model: 'Qwen/Qwen3-0.6B', dtype: 'float32', dies: 1, maxModelLen: 12288 };

let serving: Settings | undefined;
let started = 0;

/** The container's stdout and stderr; docker logs replays them on its own two streams. */
function containerLog(): string {
  const r = spawnSync('docker', ['logs', CONTAINER], { encoding: 'utf-8', maxBuffer: 256 * 1024 * 1024 });
  return `${r.stdout ?? ''}${r.stderr ?? ''}`;
}

/** Keep the server's log beside the report before its container is removed. */
function keepLog(): void {
  if (!serving) return;
  const report = resolve(arg('output') ?? 'results/report.json');
  mkdirSync(dirname(report), { recursive: true });
  writeFileSync(resolve(dirname(report), `${basename(report, '.json')}-server-${started}.log`), containerLog());
}
beforeRestore(keepLog);

/** The scripts' output goes to stderr: stdout is the report. */
function script(name: string, args: string[], env: NodeJS.ProcessEnv = process.env): void {
  const r = spawnSync('bash', [resolve(K80, name), ...args], { stdio: ['ignore', 2, 2], env });
  if (r.status !== 0) throw new Error(`${name} exited ${r.status}`);
}

/**
 * Serve the first row of the table with `change` applied. A server already up
 * with the same settings is kept; any other is replaced. The GPU hand-over is
 * runTest's: this only starts and stops the container.
 */
export async function serve(change: Partial<Settings> = {}): Promise<Settings> {
  assertInRun();
  const s: Settings = { ...FIRST_ROW, ...change };
  if (serving && JSON.stringify(serving) === JSON.stringify(s)) return s;
  if (serving) {
    keepLog();
    script('container-down.sh', []);
  }
  serving = undefined;
  script('container-up.sh', ['600'], {
    ...process.env,
    COMPOSE_FILE: process.env.COMPOSE_FILE ?? 'docker-compose.yml:docker-compose.ci.yml',
    MODEL: s.model,
    DTYPE: s.dtype,
    TP_SIZE: String(s.dies),
    MAX_MODEL_LEN: String(s.maxModelLen),
    // Two dies have hung this machine before: keep the tensor-parallel start-up trace.
    VLLM_K80_TRACE: process.env.VLLM_K80_TRACE ?? (s.dies > 1 ? '1' : '0'),
  });
  serving = s;
  started++;
  heldFixed(`${s.model}, ${s.dtype}, ${s.dies} die(s), ${s.maxModelLen}-token context`);
  return s;
}

/** Weights and KV cache size as the server logged them at start. Per die when it runs on two. */
export function serverInfo(): { weightsGib: number; kvTokens: number } {
  const log = containerLog();
  const weights = log.match(/model weights take ([\d.]+)\s*GiB/);
  const blocks = log.match(/# cuda blocks: (\d+)/);
  return { weightsGib: weights ? Number(weights[1]) : 0, kvTokens: blocks ? Number(blocks[1]) * BLOCK_TOKENS : 0 };
}

const round = (x: number, digits = 2): number => (Number.isFinite(x) ? +x.toFixed(digits) : 0);

/**
 * A model that reasons inline wraps it in <think>…</think> ahead of its answer,
 * and the server streams both as content. Split them the way Ollama does, so a
 * Reply means the same here as in ollama37: `response` is the answer, empty when
 * the budget ended inside the reasoning.
 */
function splitThinking(content: string): { response: string; thinking: string } {
  const open = content.indexOf('<think>');
  if (open < 0) return { response: content, thinking: '' };
  const close = content.indexOf('</think>', open);
  if (close < 0) return { response: content.slice(0, open), thinking: content.slice(open + 7) };
  return { response: content.slice(0, open) + content.slice(close + 8), thinking: content.slice(open + 7, close) };
}

interface Timed {
  reply: Reply;
  inTokens: number;
  outTokens: number;
  /** Milliseconds: request sent, first token, last token. */
  sent: number;
  first: number;
  last: number;
}

/**
 * One streamed chat completion. `onToken` is told how many tokens have arrived,
 * so a staggered call can start part-way through this one.
 */
function call(model: string, text: string, options: Record<string, unknown>, onToken?: (n: number) => void): Promise<Timed> {
  const body = JSON.stringify({
    model,
    messages: [{ role: 'user', content: text }],
    max_tokens: Number(options.num_predict ?? 100),
    temperature: Number(options.temperature ?? 0),
    ...(options.seed !== undefined ? { seed: Number(options.seed) } : {}),
    stream: true,
    stream_options: { include_usage: true },
  });
  return new Promise<Timed>((done, fail) => {
    const sent = performance.now();
    let first = 0, last = 0, tokens = 0, pending = '', response = '', thinking = '', doneReason = '';
    let usage: { prompt_tokens: number; completion_tokens: number } | undefined;
    const req = http.request(new URL('/v1/chat/completions', HOST), {
      method: 'POST',
      // A connection of its own: node keeps sockets alive, and one kept from a
      // server this run has since replaced hangs up on the next request.
      agent: false,
      headers: { 'content-type': 'application/json', 'content-length': String(Buffer.byteLength(body)) },
    }, (res) => {
      res.setEncoding('utf-8');
      if (res.statusCode !== 200) {
        let error = '';
        res.on('data', (c: string) => (error += c));
        res.on('end', () => fail(new Error(`HTTP ${res.statusCode}: ${error.slice(0, 300)}`)));
        return;
      }
      res.on('data', (chunk: string) => {
        pending += chunk;
        const lines = pending.split('\n');
        pending = lines.pop() ?? '';
        for (const line of lines) {
          if (!line.startsWith('data: ') || line === 'data: [DONE]') continue;
          const event = JSON.parse(line.slice(6));
          if (event.usage) usage = event.usage;
          const choice = event.choices?.[0];
          if (!choice) continue;
          if (choice.finish_reason) doneReason = choice.finish_reason;
          const content: string = choice.delta?.content ?? '';
          const reasoning: string = choice.delta?.reasoning_content ?? '';
          if (!content && !reasoning) continue;
          const now = performance.now();
          if (!first) first = now;
          last = now;
          response += content;
          thinking += reasoning;
          onToken?.(++tokens);
        }
      });
      res.on('error', fail);
      res.on('end', () => {
        if (!usage) return fail(new Error('the stream ended without a usage record'));
        const split = splitThinking(response);
        done({
          reply: { response: split.response.trim(), thinking: (thinking + split.thinking).trim(), doneReason, evalCount: usage.completion_tokens },
          inTokens: usage.prompt_tokens, outTokens: usage.completion_tokens, sent, first: first || performance.now(), last: last || performance.now(),
        });
      });
    });
    req.setTimeout(1_200_000, () => req.destroy(new Error('no response in 1200s')));
    req.on('error', fail);
    req.end(body);
  });
}

export interface GenerateOptions {
  /** How many calls send the prompt. Default 1. */
  calls?: number;
  /** `together`: all at once. `staggered`: each once the one before has streamed some tokens. */
  arrival?: 'together' | 'staggered';
}

/**
 * Send the named prompt to the server, check each reply, and record it for the
 * judge under `option`. One Result per call.
 */
export async function generate(option: string, name: string, opts: GenerateOptions = {}): Promise<Result[]> {
  assertInRun();
  if (!serving) throw new Error('generate() needs a server: call serve() first');
  const { model } = serving;
  const p = prompt(name);
  const count = opts.calls ?? 1;
  const staggered = opts.arrival === 'staggered';

  const running: Promise<Timed>[] = [];
  /** release[i] lets call i+1 go; call i fires it at its STAGGER_AFTER-th token. */
  const release: (() => void)[] = [];
  for (let i = 0; i < count; i++) {
    if (staggered && i > 0) {
      // Wait until the call before this one is decoding, or has ended early.
      await new Promise<void>((go) => {
        release[i - 1] = go;
        running[i - 1].then(() => go(), () => go());
      });
    }
    running.push(call(model, p.text, p.options, (n) => { if (n === STAGGER_AFTER) release[i]?.(); }));
  }
  const timed = await Promise.all(running);

  // The group's decode rate: tokens after each call's first, over the time from
  // the first token of the group to its last.
  const span = (Math.max(...timed.map((t) => t.last)) - Math.min(...timed.map((t) => t.first))) / 1000;
  const totalDecodeTps = round(timed.reduce((n, t) => n + Math.max(t.outTokens - 1, 0), 0) / span);

  return timed.map((t, i) =>
    record({
      option,
      model,
      prompt: name,
      judgeName: p.judgeName,
      judgePass: p.judgePass,
      judgeTemplate: p.judge,
      reply: t.reply,
      check: check(t.reply, p.expect, p.reject),
      metrics: {
        inTokens: t.inTokens,
        outTokens: t.outTokens,
        prefillTps: round(t.inTokens / ((t.first - t.sent) / 1000)),
        decodeTps: round(Math.max(t.outTokens - 1, 0) / ((t.last - t.first) / 1000)),
        firstTokenS: round((t.first - t.sent) / 1000),
        // A test about calls reports the group's rate on every row, the 1-call row included.
        ...(opts.calls !== undefined ? { call: i + 1, totalDecodeTps, ...(staggered && i > 0 ? { sentAfterTokens: STAGGER_AFTER } : {}) } : {}),
      },
    }),
  );
}

/**
 * The model test, as ollama37's models.ts: does the server just started load the
 * model and answer in language? Its row also carries what the server holds:
 * weights, KV cache size and VRAM per die.
 */
export async function modelTest(option: string): Promise<Result> {
  const [r] = await generate(option, 'short-answer');
  const info = serverInfo();
  const vram = serverVram();
  r.checks.resident = { pass: vram.totalMib > 0, reason: `${vram.totalMib} MiB across ${vram.dies} die(s)` };
  r.metrics.weightsGib = info.weightsGib;
  r.metrics.kvTokens = info.kvTokens;
  r.metrics.vramMib = `${vram.totalMib}/${vram.dies}die`;
  return r;
}
