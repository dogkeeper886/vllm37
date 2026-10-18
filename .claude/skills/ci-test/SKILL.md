---
description: >-
  CI tests vllm37 on the self-hosted K80 runner with manual workflows,
  test-*.yml, whose steps are the test and test-end actions: five that test one
  condition each on the baseline model, test-all.yml, which runs the five in
  sequence, and test-use-case.yml, which serves each model profile once. A run
  stops the vllm37 and ollama services, serves vllm37, sends short-answer and
  the timed requests, starts the services again and has a judge read every
  reply. An agent wants a test result and curls a server it started by hand.
  The hand test skips the judge, so its pass says nothing about whether the
  replies were language. This skill asks the agent to test vllm37 with CI by
  the steps below.
---

# Test vllm37 with CI

1. Pick the workflow: `test-all.yml` to check a change, one test: `test-concurrent-calls.yml`, `test-arrival.yml`, `test-request-length.yml`, `test-die-count.yml` or `test-weight-dtype.yml`, or `test-use-case.yml` to check that each supported model still serves and answers.
2. `gh workflow run <file> -R dogkeeper886/vllm37 --ref <branch>`, `main` or the branch under test.
3. Leave `model` empty and `TEST_MODEL` at the baseline, `Qwen/Qwen3-0.6B`, to check a change: the small, widely supported model serves on one die in float32, so a failure points at the change, not the model; pass `-f model=<id>` only to test that model's port.
4. To support a new model, add its profile to `docker/k80/models/`, vLLM's flags without the leading `--`; `test-use-case.yml` runs every profile, so no script changes.
5. To change a value for this run, add `-f <input>=<value>`; the workflow's `on.workflow_dispatch.inputs` names each input and its fallback.
6. To change the standing values, set the repository variable: `gh variable list -R dogkeeper886/vllm37` shows `TEST_*`, and `gh variable set <name> -R dogkeeper886/vllm37 --body <value>` changes one.
7. `gh run list -R dogkeeper886/vllm37 --workflow <file> --limit 1`
8. `gh run watch <run-id> -R dogkeeper886/vllm37 --exit-status`
9. `gh run download <run-id> -R dogkeeper886/vllm37`, then read `report.json` and `server.json`; a `test-all.yml` run has one folder per test job and the summary table on the run page. Each row's `dies` holds every die's clock and peak temperature; a tok/s with `dies.valid: false`, struck through in the tables, measured a hot or throttled die, so quote it only with its `dies.reason`.
10. Dispatch the next test after this run ends: the lock keeps one waiting run and cancels an older one.
