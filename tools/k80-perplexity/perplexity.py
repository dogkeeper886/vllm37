#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Perplexity of a served model on a fixed text, with checks of the
measurement itself (#152).

A client of a running vLLM server: the model, its flags and its dies are the
server's, so every profile is measured the same way and a model added later
needs no change here. run.sh starts the host build's server for a profile and
calls `measure`.

  perplexity.py measure [--url URL] [--text NAME|PATH] [--out FILE]
      scores the text in windows, then runs the checks; writes one result
  perplexity.py gate RESULT
      the result against the calibration in criteria.yaml: pass or fail
  perplexity.py rank RESULT...
      results of one model's checkpoints, the lowest perplexity first

Perplexity is exp(-mean log p) over every token of each window but the first,
p being the probability the model gives the text's next token (prompt
log-probabilities from /v1/completions). The server builds them on one die,
outside the memory vLLM plans for: about six float32 [tokens, vocabulary]
tensors when the logits are gathered from several dies, 1.5 GiB for 256 tokens
of a 248k vocabulary, while 512 tokens ran out of memory at TP=2. So every
request stays within one window, and run.sh serves with a smaller memory
budget. The checks:

  repeat    window 1 scored twice: the largest log p difference
  shuffled  window 1's tokens in random order: its perplexity
  copied    the first half of window 1 twice: the second copy's perplexity
  decode    after each of several prompts, greedy tokens scored as they are
            written (decode) and again as a prompt (prefill): the largest log
            p difference, and the smallest of those over the prompts
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import random
import sys
import urllib.error
import urllib.request

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
NEUTRAL = {
    "temperature": 0,
    "repetition_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
}


def call(url, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url + path, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        sys.exit(f"perplexity.py: {path}: HTTP {e.code}: "
                 f"{e.read().decode()[:500]}")


def score(url, model, ids, start):
    out = call(url, "/v1/completions", {
        "model": model,
        "prompt": ids,
        "max_tokens": 1,
        "prompt_logprobs": 0,
        **NEUTRAL
    })
    entries = out["choices"][0]["prompt_logprobs"]
    return [entries[i][str(ids[i])]["logprob"] for i in range(start, len(ids))]


def perplexity(lps):
    return math.exp(-sum(lps) / len(lps))


def gap(a, b):
    return max(abs(x - y) for x, y in zip(a, b, strict=True))


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def measure(args):
    served = call(args.url, "/v1/models")["data"][0]
    model, max_len = served["id"], served.get("max_model_len")
    path = args.text
    if os.sep not in path and not path.endswith(".txt"):
        path = os.path.join(HERE, "texts", path + ".txt")
    with open(path, "rb") as f:
        raw = f.read()

    def tokenize(text, special):
        return call(args.url, "/tokenize", {
            "model": model,
            "prompt": text,
            "add_special_tokens": special
        })["tokens"]

    ids = tokenize(raw.decode("utf-8"), False)
    prefix = tokenize("", True)
    n, w = args.windows, args.window_tokens
    if len(ids) < n * w:
        sys.exit(f"perplexity.py: {path} is {len(ids)} tokens, fewer than "
                 f"{n} windows of {w}")
    if args.decode_prompt + args.decode_tokens > w:
        sys.exit("perplexity.py: the decode check's prompt and tokens exceed "
                 f"one window of {w}")
    if args.decode_prompts > n:
        sys.exit(f"perplexity.py: {args.decode_prompts} decode prompts need "
                 f"as many windows; there are {n}")
    need = len(prefix) + w
    if max_len and need > max_len:
        sys.exit(f"perplexity.py: {need} tokens do not fit the server's "
                 f"{max_len}-token context")
    windows = [ids[i * w:(i + 1) * w] for i in range(n)]
    start = len(prefix) + 1

    def log(what):
        print(f"perplexity.py: {what}", file=sys.stderr, flush=True)

    scored = []
    for i, window in enumerate(windows):
        scored.append(score(args.url, model, prefix + window, start))
        log(f"window {i + 1}/{n}: {perplexity(scored[-1]):.4f}")
    lps = [lp for s in scored for lp in s]

    again = score(args.url, model, prefix + windows[0], start)
    shuffled = list(windows[0])
    random.Random(args.seed).shuffle(shuffled)
    shuffled_lps = score(args.url, model, prefix + shuffled, start)
    half = windows[0][:w // 2]
    copied_lps = score(args.url, model, prefix + half + half,
                       len(prefix) + len(half))

    decode_gaps = []
    for i in range(args.decode_prompts):
        prompt = prefix + windows[i][:args.decode_prompt]
        out = call(
            args.url, "/v1/completions", {
                "model": model,
                "prompt": prompt,
                "max_tokens": args.decode_tokens,
                "logprobs": 0,
                "ignore_eos": True,
                "return_tokens_as_token_ids": True,
                **NEUTRAL
            })["choices"][0]["logprobs"]
        written = [int(t.split(":", 1)[1]) for t in out["tokens"]]
        prefilled = score(args.url, model, prompt + written, len(prompt))
        decode_gaps.append(gap(out["token_logprobs"], prefilled))
        log(f"decode prompt {i + 1}/{args.decode_prompts}: "
            f"gap {decode_gaps[-1]:.3g}")

    now = datetime.datetime.now().astimezone()
    result = {
        "model": model,
        "maxModelLen": max_len,
        "label": args.label,
        "build": args.build,
        "server": args.url,
        "date": now.isoformat(timespec="seconds"),
        "text": os.path.basename(path),
        "textSha256": sha256(raw),
        "tokenIdsSha256": sha256(json.dumps(prefix + ids[:n * w]).encode()),
        "windows": n,
        "windowTokens": w,
        "scoredTokens": len(lps),
        "perplexity": perplexity(lps),
        "windowPerplexity": [perplexity(s) for s in scored],
        "repeatGap": gap(scored[0], again),
        "shuffledPerplexity": perplexity(shuffled_lps),
        "copiedPerplexity": perplexity(copied_lps),
        "decodePrompts": args.decode_prompts,
        "decodeTokens": len(written),
        "decodeGaps": decode_gaps,
        "decodeGap": min(decode_gaps),
        "seed": args.seed,
    }
    text = json.dumps(result, indent=1)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        log(f"wrote {args.out}")
    print(text)


def load(path):
    with open(path) as f:
        return json.load(f)


def criteria():
    with open(os.path.join(HERE, "criteria.yaml")) as f:
        return yaml.safe_load(f)


def table(head, rows):
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for row in rows:
        print("| " + " | ".join(str(c) for c in row) + " |")


def gate(args):
    c = criteria()
    cal_path = args.calibration or (c["calibration"] and os.path.join(
        HERE, c["calibration"]))
    if not cal_path:
        sys.exit("perplexity.py gate: criteria.yaml names no calibration yet; "
                 "record one first (README.md)")
    r, cal = load(args.result), load(cal_path)
    for path, res in ((args.result, r), (cal_path, cal)):
        if "decodePrompts" not in res:
            sys.exit(f"perplexity.py gate: {path} ran the decode check on "
                     "one prompt; measure it again")
    limit = c["decode_gap_factor"] * cal["decodeGap"]
    ratio = r["shuffledPerplexity"] / r["perplexity"]
    checks = [
        (f"prefill/decode gap, the smallest of {r['decodePrompts']} prompts",
         f"{r['decodeGap']:.3g}", f"≤ {limit:.3g}", r["decodeGap"] <= limit),
        ("copied", f"{r['copiedPerplexity']:.4f}", f"< {c['copy_max']}",
         r["copiedPerplexity"] < c["copy_max"]),
        ("shuffled / text", f"{ratio:.1f}", f"> {c['shuffle_min_ratio']}",
         ratio > c["shuffle_min_ratio"]),
    ]
    if r["tokenIdsSha256"] == cal["tokenIdsSha256"]:
        checks.append(("text", f"{r['perplexity']:.4f}",
                       f"< {cal['perplexity']:.4f}, the calibration's",
                       r["perplexity"] < cal["perplexity"]))
    else:
        checks.append(("text", f"{r['perplexity']:.4f}",
                       "not compared: token IDs differ from the calibration's",
                       None))
    print(f"{r['label'] or r['model']} against {cal['label'] or cal['model']}")
    print()
    verdict = {True: "pass", False: "FAIL", None: "—"}
    table(["check", "value", "must be", ""],
          [(n, v, m, verdict[ok]) for n, v, m, ok in checks])
    sys.exit(1 if any(ok is False for *_, ok in checks) else 0)


def rank(args):
    tie = criteria()["tie"]
    results = sorted((load(p) for p in args.results),
                     key=lambda r: r["perplexity"])
    if len({r["tokenIdsSha256"] for r in results}) > 1:
        sys.exit("perplexity.py rank: the results scored different token IDs "
                 "(another text or tokenizer), so they do not compare")
    best = results[0]["perplexity"]
    rows = []
    for i, r in enumerate(results):
        rel = r["perplexity"] / best - 1
        note = ("tie" if i and tie is not None and rel < tie else "")
        rows.append((i + 1, r["label"] or r["model"], r["build"] or "",
                     f"{r['perplexity']:.4f}", f"+{rel:.2%}", note))
    table(["rank", "checkpoint", "build", "perplexity", "vs first", ""], rows)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure", help="score the text and run the checks")
    # a server's base URL (default http://localhost:8000)
    m.add_argument("--url", default="http://localhost:8000")
    m.add_argument("--text", default="log-of-a-cowboy",
                   help="a name in texts/, or a path")
    # a whole number from 1; times --window-tokens, at most the text's tokens
    # (default 32)
    m.add_argument("--windows", type=int, default=32)
    # a whole number from 2; with the tokenizer's prefix, at most the server's
    # max_model_len (default 256)
    m.add_argument("--window-tokens", type=int, default=256)
    # a whole number from 1, at most --windows (default 6)
    m.add_argument("--decode-prompts", type=int, default=6)
    m.add_argument("--decode-prompt", type=int, default=128,
                   help="prompt tokens before the decode check's greedy tokens")
    # a whole number from 1; with --decode-prompt, at most --window-tokens
    # (default 128)
    m.add_argument("--decode-tokens", type=int, default=128)
    # an integer (default 0)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--label", default="", help="such as the profile name")
    m.add_argument("--build", default="", help="such as the server's commit")
    m.add_argument("--out", help="also write the result here")
    g = sub.add_parser("gate", help="a result against the calibration")
    # a measure result's path
    g.add_argument("result")
    g.add_argument("--calibration",
                   help="a result to compare with, instead of criteria.yaml's")
    r = sub.add_parser("rank", help="results of one model's checkpoints")
    # measure results' paths, one or more, of one text and tokenizer
    r.add_argument("results", nargs="+")
    args = p.parse_args()
    {"measure": measure, "gate": gate, "rank": rank}[args.cmd](args)


if __name__ == "__main__":
    main()
