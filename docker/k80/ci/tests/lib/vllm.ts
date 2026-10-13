/**
 * The only code that talks to the server under test, through the OpenAI client.
 * Takes the place of ollama37's cicd/tests/lib/ollama.ts.
 *
 *   generate('calls=2', 'speech-rewrite', { calls: 2 })
 *     └─ prompts.yaml['speech-rewrite'] → text, options, judge question
 *          └─ chat.completions.create, streamed → reply, prefill and decode speed
 *               └─ check(reply)
 *                    └─ recorded for the judge, who runs after the GPU is back
 *                         └─ Result { reply, check, judge, pass }
 *
 * `generate` is the only export that returns model text, and it only runs inside
 * runTest, so every reply is checked and recorded for the judge. Starting and
 * stopping the server is not done here: that is docker/k80/ci/service-start.sh
 * and service-stop.sh.
 */
import OpenAI from 'openai';
import { check, type Reply } from './check.js';
import { prompt } from './prompts.js';
import { assertInRun, record, type Result } from './run.js';

export const HOST = process.env.VLLM37_HOST ?? 'http://localhost:8000';

// No retries: a call sent twice would be timed twice and would change how many
// calls the server holds at once.
const client = new OpenAI({ baseURL: `${HOST}/v1`, apiKey: 'none', maxRetries: 0, timeout: 1_200_000 });

/** What the server says it serves. */
export interface Served {
  model: string;
  /** Context length in tokens; 0 when the server does not report one. */
  maxModelLen: number;
}

let serving: Promise<Served> | undefined;

/** The model the server serves and its context length, asked once per run. */
export function served(): Promise<Served> {
  serving ??= client.models.list().then(({ data }) => {
    const m = data[0] as (OpenAI.Model & { max_model_len?: number }) | undefined;
    if (!m) throw new Error(`${HOST} serves no model`);
    return { model: m.id, maxModelLen: m.max_model_len ?? 0 };
  });
  return serving;
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
async function call(model: string, text: string, options: Record<string, unknown>, onToken?: (n: number) => void): Promise<Timed> {
  const sent = performance.now();
  const stream = await client.chat.completions.create({
    model,
    messages: [{ role: 'user', content: text }],
    max_tokens: Number(options.num_predict ?? 100),
    temperature: Number(options.temperature ?? 0),
    ...(options.seed !== undefined ? { seed: Number(options.seed) } : {}),
    stream: true,
    stream_options: { include_usage: true },
  });
  let first = 0, last = 0, tokens = 0, response = '', thinking = '', doneReason = '';
  let usage: OpenAI.CompletionUsage | undefined;
  for await (const chunk of stream) {
    if (chunk.usage) usage = chunk.usage;
    const choice = chunk.choices[0];
    if (!choice) continue;
    if (choice.finish_reason) doneReason = choice.finish_reason;
    const content = choice.delta?.content ?? '';
    // vLLM's field for reasoning a parser has split off; not in the OpenAI types.
    const reasoning = (choice.delta as { reasoning_content?: string } | undefined)?.reasoning_content ?? '';
    if (!content && !reasoning) continue;
    const now = performance.now();
    if (!first) first = now;
    last = now;
    response += content;
    thinking += reasoning;
    onToken?.(++tokens);
  }
  if (!usage) throw new Error('the stream ended without a usage record');
  const split = splitThinking(response);
  const end = performance.now();
  return {
    reply: { response: split.response.trim(), thinking: (thinking + split.thinking).trim(), doneReason, evalCount: usage.completion_tokens },
    inTokens: usage.prompt_tokens, outTokens: usage.completion_tokens, sent, first: first || end, last: last || end,
  };
}

export interface GenerateOptions {
  /** How many calls send the prompt. Default 1. */
  calls?: number;
  /**
   * Tokens a call has streamed before the next one is sent. 0, the default,
   * sends all calls together.
   */
  stagger?: number;
}

/**
 * Send the named prompt to the server, check each reply, and record it for the
 * judge under `option`. One Result per call.
 */
export async function generate(option: string, name: string, opts: GenerateOptions = {}): Promise<Result[]> {
  assertInRun();
  const { model } = await served();
  const p = prompt(name);
  const count = opts.calls ?? 1;
  const stagger = opts.stagger ?? 0;

  const running: Promise<Timed>[] = [];
  /** release[i] lets call i+1 go; call i fires it at its `stagger`-th token. */
  const release: (() => void)[] = [];
  for (let i = 0; i < count; i++) {
    if (stagger > 0 && i > 0) {
      // Wait until the call before this one is decoding, or has ended early.
      await new Promise<void>((go) => {
        release[i - 1] = go;
        running[i - 1].then(() => go(), () => go());
      });
    }
    running.push(call(model, p.text, p.options, (n) => { if (n === stagger) release[i]?.(); }));
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
        totalDecodeTps,
        ...(count > 1 ? { call: i + 1 } : {}),
        ...(stagger > 0 && i > 0 ? { sentAfterTokens: stagger } : {}),
      },
    }),
  );
}
