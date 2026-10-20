# K80 perplexity

Measures a served model's perplexity on a fixed text, and checks the measurement itself, so a port can show its numerics hold on the K80 (#152). It runs by hand while porting; no CI workflow calls it, and its numbers go in the port's pull request.

The tool is a client of the vLLM server. The model, its flags and its dies come from the server, which `run.sh` starts from a profile. So every model is measured the same way, and a model added later needs only its profile.

## Measure a profile

```bash
tools/k80-perplexity/run.sh qwen3.5-0.8b                      # a name in docker/k80/models/
tools/k80-perplexity/run.sh path/to/profile.yaml              # or a path
CUDA_VISIBLE_DEVICES=2 tools/k80-perplexity/run.sh qwen3.5-0.8b
```

`run.sh` starts `tools/k80-host/serve.sh` with the profile on port 8011 and the first tensor × pipeline parallel dies, waits until it answers, measures, and stops it. It refuses a die another process holds. The result and the server log go to `~/.local/state/k80-perplexity/` (`RESULTS`).

Against a server already running, such as the docker image:

```bash
tools/k80-perplexity/perplexity.py measure --url http://localhost:8000 --label <name> --out result.json
```

## What it measures

The text is tokenized once and cut into 32 windows of 256 tokens. Each window goes to `/v1/completions` as token IDs with `prompt_logprobs=0`, which returns the log-probability p the model gives each of the text's next tokens. Perplexity is exp(−mean log p) over every token of each window but the first. The requests are greedy with every penalty off, so a model's generation config can't change a score.

| Check | Input | A sound result |
|---|---|---|
| `repeatGap` | window 1 scored twice | 0 |
| `shuffledPerplexity` | window 1's tokens in random order | far above the text's |
| `copiedPerplexity` | the first half of window 1, twice; the second copy scored | near 1 |
| `decodeGap` | after each of six prompts, the first 128 tokens of a window: 128 greedy tokens scored as they are written (decode), then again as a prompt (prefill); the smallest of the six largest gaps (`decodeGaps`) | small: the two paths are separate code in a port, so a bug in either gaps on every prompt. An MoE model can gap on some prompts with no bug: where two experts tie for the router's last pick, the two paths pick different ones |

Two results compare only when they scored the same token IDs (`tokenIdsSha256`): the same text and the same tokenizer.

**Why 256 tokens, and a 0.75 memory budget:** the server builds the prompt log-probabilities on one die, outside the memory vLLM plans for: about six float32 [tokens × vocabulary] tensors when the logits are gathered from several dies. For a 248k vocabulary that is 1.5 GiB at 256 tokens; 2,048 tokens ran out of memory on one die, and 512 at TP=2. `run.sh` serves with `--gpu-memory-utilization 0.75` (`GPU_MEM_UTIL`), over the profile's own, to leave that room. It sizes the KV cache only, so no score depends on it.

## Judge a checkpoint

```bash
tools/k80-perplexity/perplexity.py gate result.json        # against the calibration
tools/k80-perplexity/perplexity.py rank a.json b.json ...  # one model's checkpoints
```

`criteria.yaml` holds the thresholds and names the calibration: a result of an unquantized model, which the gate compares each checkpoint with.

- **gate:** `decodeGap` ≤ `decode_gap_factor` × the calibration's; `copiedPerplexity` < `copy_max`; `shuffledPerplexity` > `shuffle_min_ratio` × the text's; and, when the token IDs match the calibration's, a perplexity below the calibration's. It exits 1 on a failed check.
- **rank:** the lowest perplexity first. A checkpoint within `tie` of the first ties with it.

The calibration is Qwen3.5-0.8B in FP32 on one die (`calibration/qwen3.5-0.8b.json`):

| | Qwen3.5-0.8B, TP=1 | Qwen3.5-0.8B, TP=2 | Qwen3.5-35B-A3B GPTQ int4, TP=4 | Qwen3.6-35B-A3B GPTQ int4, TP=4 |
|---|---|---|---|---|
| perplexity | 40.966 | 40.966 (1.7e-7 apart) | 16.938 | 17.559 |
| `repeatGap` | 0 | 0 | 0 | 0 |
| shuffled / text | 63.7 | 63.7 | 91.6 | 87.1 |
| `copiedPerplexity` | 1.154 | 1.154 | 1.189 | 1.182 |
| `decodeGap` | 1.25e-5 | 1.22e-5 | 1.96e-5 | 2.65e-5 |
| `decodeGaps` over 1e-3 | none | none | 2 of 6 | 3 of 6 |

## Files

| File | What it holds |
|---|---|
| `run.sh` | the host-build run for one profile |
| `perplexity.py` | `measure`, `gate` and `rank` |
| `criteria.yaml` | the thresholds and the calibration's path |
| `calibration/` | the calibration result the gate compares with |
| `texts/` | the texts, with their sources and licences in `SOURCES.md` |
