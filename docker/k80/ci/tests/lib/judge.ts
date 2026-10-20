/**
 *   JUDGE_BASE_URL  an ollama server's URL, or unset for Claude
 *   JUDGE_MODEL     a model that server lists, or unset for the agent's own
 *   JUDGE_AGENT     a shell command that starts an ACP agent on stdio, or unset for
 *                   the bundled Claude agent
 */
import { spawn, type ChildProcess } from 'node:child_process';
import { Readable, Writable } from 'node:stream';
import { createRequire } from 'node:module';
import { dirname, join, resolve } from 'node:path';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import {
  ClientSideConnection,
  ndJsonStream,
  PROTOCOL_VERSION,
  type Client,
  type SessionNotification,
  type RequestPermissionRequest,
  type RequestPermissionResponse,
} from '@agentclientprotocol/sdk';
import { warmupQuestion } from './prompts.js';

export interface Judgment {
  verdict: 'yes' | 'no' | 'abstain';
  reason: string;
  thinking?: string;
}

// milliseconds, whole numbers
const TIMEOUT_MS = 300_000;
const WARMUP_MS = 900_000;
const LOOP_WINDOW_MS = 30_000;
const LOOP_CHECK_MS = 5_000;
// repeats of one 3-word phrase, a whole number
const LOOP_LIMIT = 100;
// characters, whole numbers
const REPLY_LIMIT = 1000;
const RESULT_LIMIT = 2500;

const JUDGE_CWD = mkdtempSync(join(tmpdir(), 'vllm37-judge-'));

function agentEnv(): NodeJS.ProcessEnv {
  const env = { ...process.env };
  delete env.CLAUDECODE;
  delete env.CLAUDE_CODE_ENTRYPOINT;
  delete env.CLAUDE_CODE_SSE_PORT;
  if (env.JUDGE_BASE_URL) {
    env.ANTHROPIC_BASE_URL = env.JUDGE_BASE_URL;
    env.ANTHROPIC_AUTH_TOKEN = 'ollama';
    if (env.JUDGE_MODEL) {
      env.ANTHROPIC_MODEL = env.JUDGE_MODEL;
      env.ANTHROPIC_DEFAULT_HAIKU_MODEL = env.JUDGE_MODEL;
    }
  }
  return env;
}

class Agent {
  private child?: ChildProcess;
  private conn?: ClientSideConnection;
  private messages: { id: string | undefined; text: string }[] = [];
  thoughts = '';
  private streamed: { at: number; text: string }[] = [];
  private loopReason?: string;

  limit = TIMEOUT_MS;

  private async within<T>(p: Promise<T>, label: string): Promise<T> {
    const ms = this.limit;
    let timer: NodeJS.Timeout;
    const t = new Promise<never>((_, rej) => { timer = setTimeout(() => rej(new Error(`${label} exceeded ${ms}ms`)), ms); });
    try { return await Promise.race([p, t]); } finally { clearTimeout(timer!); }
  }

  private spawnAgent(): ChildProcess {
    const stdio: ['pipe', 'pipe', 'inherit'] = ['pipe', 'pipe', 'inherit'];
    if (process.env.JUDGE_AGENT) return spawn(process.env.JUDGE_AGENT, { cwd: JUDGE_CWD, stdio, env: agentEnv(), shell: true });
    const req = createRequire(import.meta.url);
    const entry = resolve(dirname(req.resolve('@agentclientprotocol/claude-agent-acp/package.json')), 'dist/index.js');
    return spawn(process.execPath, [entry], { cwd: JUDGE_CWD, stdio, env: agentEnv() });
  }

  private async start(): Promise<void> {
    if (this.conn) return;
    this.child = this.spawnAgent();
    const stream = ndJsonStream(
      Writable.toWeb(this.child.stdin!) as WritableStream<Uint8Array>,
      Readable.toWeb(this.child.stdout!) as ReadableStream<Uint8Array>,
    );
    const client: Client = {
      sessionUpdate: async (params: SessionNotification) => {
        const u = params.update;
        if ((u.sessionUpdate === 'agent_message_chunk' || u.sessionUpdate === 'agent_thought_chunk') && u.content.type === 'text') {
          this.streamed.push({ at: Date.now(), text: u.content.text });
          if (u.sessionUpdate === 'agent_thought_chunk') this.thoughts += u.content.text;
          if (u.sessionUpdate === 'agent_message_chunk') {
            const id = (u as { messageId?: string | null }).messageId ?? undefined;
            const last = this.messages[this.messages.length - 1];
            if (last && last.id === id) last.text += u.content.text;
            else this.messages.push({ id, text: u.content.text });
          }
        }
      },
      requestPermission: async (p: RequestPermissionRequest): Promise<RequestPermissionResponse> => {
        const reject = p.options.find((o) => o.kind?.startsWith('reject'));
        return reject ? { outcome: { outcome: 'selected', optionId: reject.optionId } } : { outcome: { outcome: 'cancelled' } };
      },
      readTextFile: async () => { throw new Error('the judge has no filesystem'); },
      writeTextFile: async () => { throw new Error('the judge has no filesystem'); },
    };
    this.conn = new ClientSideConnection(() => client, stream);
    try {
      await this.within(this.conn.initialize({
        protocolVersion: PROTOCOL_VERSION,
        clientCapabilities: { fs: { readTextFile: false, writeTextFile: false }, terminal: false },
        clientInfo: { name: 'vllm37-judge', version: '3.0.0' },
      }), 'agent initialize');
    } catch (e) { this.kill(); throw e; }
  }

  private checkForLoop(sessionId: string): void {
    if (this.loopReason || !this.conn) return;
    const since = Date.now() - LOOP_WINDOW_MS;
    this.streamed = this.streamed.filter((c) => c.at >= since);
    const words = this.streamed.map((c) => c.text).join('').toLowerCase().split(/\s+/).filter(Boolean);
    const counts = new Map<string, number>();
    let top = '', topN = 0;
    for (let i = 0; i + 2 < words.length; i++) {
      const p = `${words[i]} ${words[i + 1]} ${words[i + 2]}`;
      const n = (counts.get(p) ?? 0) + 1;
      counts.set(p, n);
      if (n > topN) { top = p; topN = n; }
    }
    if (topN < LOOP_LIMIT) return;
    this.loopReason = `judge looped on "${top.slice(0, 40)}"`;
    this.conn.cancel({ sessionId }).catch(() => {});
  }

  async ask(question: string): Promise<string> {
    await this.start();
    const { sessionId } = await this.within(
      this.conn!.newSession({ cwd: JUDGE_CWD, mcpServers: [], _meta: { claudeCode: { options: { tools: [], strictMcpConfig: true } } } }),
      'agent session/new',
    );
    this.messages = [];
    this.streamed = [];
    this.thoughts = '';
    this.loopReason = undefined;
    const guard = setInterval(() => this.checkForLoop(sessionId), LOOP_CHECK_MS);
    try {
      await this.within(this.conn!.prompt({ sessionId, prompt: [{ type: 'text', text: question }] }), 'agent turn');
    } catch (e) {
      this.kill();
      throw this.loopReason ? new Error(this.loopReason) : e;
    } finally {
      clearInterval(guard);
    }
    if (this.loopReason) { this.kill(); throw new Error(this.loopReason); }
    this.conn?.closeSession({ sessionId }).catch(() => {});
    return this.messages.map((m) => m.text).join('\n');
  }

  kill(): void {
    try { this.child?.kill('SIGKILL'); } catch {}
    this.child = undefined;
    this.conn = undefined;
  }
}

let agent: Agent | undefined;

export async function judge(template: string, reply: string, result = ''): Promise<Judgment> {
  const cut = (s: string, limit: number) =>
    s.length > limit ? `${s.slice(0, limit)}... (cut here; the full text is ${s.length} characters)` : s;
  const question = template.replaceAll('{reply}', cut(reply, REPLY_LIMIT)).replaceAll('{result}', cut(result, RESULT_LIMIT));
  if (!agent) {
    agent = new Agent();
    agent.limit = WARMUP_MS;
    await agent.ask(warmupQuestion()).catch(() => {});
    agent.limit = TIMEOUT_MS;
  }
  let answer: string;
  try {
    answer = await agent.ask(question);
  } catch (e) {
    return { verdict: 'abstain', reason: `judge could not answer: ${e instanceof Error ? e.message : e}` };
  }
  const thinking = agent.thoughts.trim();
  const first = answer.toLowerCase().replace(/^[\s*_`#>"'“”‘’]+/, '').match(/^(yes|no)(?:\1)?\b/);
  if (!first) return { verdict: 'abstain', reason: `no yes/no in: ${answer.slice(0, 120)}`, thinking };
  const v = first[1] as 'yes' | 'no';
  return { verdict: v, reason: `judge said ${v}`, thinking };
}

export async function judgeReady(waitMs = 0): Promise<void> {
  const base = process.env.JUDGE_BASE_URL;
  if (!base) return;
  const model = process.env.JUDGE_MODEL;
  const deadline = Date.now() + waitMs;
  for (;;) {
    let why: string;
    try {
      const res = await fetch(new URL('/api/tags', base), { signal: AbortSignal.timeout(5000) });
      const names = ((await res.json()) as { models?: { name: string }[] }).models?.map((m) => m.name) ?? [];
      if (!model || names.includes(model)) return;
      why = `${base} has no model ${model}`;
    } catch (e) {
      why = `no judge server at ${base}: ${e instanceof Error ? e.message : e}`;
    }
    if (Date.now() >= deadline) throw new Error(why);
    await new Promise((r) => setTimeout(r, 2000));
  }
}

export function closeJudge(): void {
  agent?.kill();
  agent = undefined;
}
