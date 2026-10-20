import OpenAI from 'openai';
import { check, type Reply } from './check.js';
import { prompt } from './prompts.js';
import { assertInRun, record, type Result } from './run.js';

// VLLM37_HOST: the server's URL without /v1 (default http://localhost:8000)
export const HOST = process.env.VLLM37_HOST ?? 'http://localhost:8000';

// maxRetries: a whole number; timeout: milliseconds per request, a whole number
const client = new OpenAI({ baseURL: `${HOST}/v1`, apiKey: 'none', maxRetries: 0, timeout: 1_200_000 });

export interface Served {
  model: string;
  maxModelLen: number;
}

let serving: Promise<Served> | undefined;

export function served(): Promise<Served> {
  serving ??= client.models.list().then(({ data }) => {
    const m = data[0] as (OpenAI.Model & { max_model_len?: number }) | undefined;
    if (!m) throw new Error(`${HOST} serves no model`);
    return { model: m.id, maxModelLen: m.max_model_len ?? 0 };
  });
  return serving;
}

const round = (x: number, digits = 2): number => (Number.isFinite(x) ? +x.toFixed(digits) : 0);

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
  sent: number;
  first: number;
  last: number;
}

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
  calls?: number;
  stagger?: number;
}

export async function generate(option: string, name: string, opts: GenerateOptions = {}): Promise<Result[]> {
  assertInRun();
  const { model } = await served();
  const p = prompt(name);
  const count = opts.calls ?? 1;
  const stagger = opts.stagger ?? 0;

  const running: Promise<Timed>[] = [];
  const release: (() => void)[] = [];
  for (let i = 0; i < count; i++) {
    if (stagger > 0 && i > 0) {
      await new Promise<void>((go) => {
        release[i - 1] = go;
        running[i - 1].then(() => go(), () => go());
      });
    }
    running.push(call(model, p.text, p.options, (n) => { if (n === stagger) release[i]?.(); }));
  }
  const timed = await Promise.all(running);

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
