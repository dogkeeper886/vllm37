export interface Reply {
  response: string;
  thinking: string;
  doneReason: string;
  evalCount: number;
}

export interface Verdict {
  pass: boolean;
  reason: string;
}

export const readable = (r: Reply): string => r.response.trim() || r.thinking.trim();

function isLoop(s: string): boolean {
  // 50: longest unit and 100: shortest reply, in characters; 5: fewest repeats; all whole numbers
  for (let u = 1; u <= 50; u++) {
    if (s.length < Math.max(100, 5 * u)) continue;
    if (s.slice(0, u).repeat(Math.ceil(s.length / u) + 1).slice(0, s.length) === s) return true;
  }
  return false;
}

export function check(r: Reply, expect?: string, reject: string[] = []): Verdict {
  const text = readable(r);
  if (!text) return { pass: false, reason: 'no text in response or thinking' };
  if (!/[\p{L}\p{N}]/u.test(text)) return { pass: false, reason: 'no letters or digits' };
  if (isLoop(text)) return { pass: false, reason: 'one short unit repeated to fill the reply' };
  const said = r.response.toLowerCase();
  const refused = reject.find((p) => said.includes(p.toLowerCase()));
  if (refused) return { pass: false, reason: `refused: "${refused}"` };
  if (expect && !r.response.includes(expect) && !r.thinking.includes(expect)) {
    return { pass: false, reason: `expected "${expect}" is missing` };
  }
  return { pass: true, reason: expect ? `has text; contains "${expect}"` : 'has text' };
}
