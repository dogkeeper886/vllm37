---
description: >-
  CI tests vllm37 on the self-hosted K80 runner with manual workflows,
  test-*.yml, whose steps are the test and test-end actions: five that test one
  condition each, and test-all.yml, which runs the five in sequence. A run
  stops the vllm37 and ollama services, serves vllm37, sends short-answer and
  the timed requests, starts the services again and has a judge read every
  reply. An agent wants a test result and curls a server it started by hand.
  The hand test skips the judge, so its pass says nothing about whether the
  replies were language. This skill asks the agent to test vllm37 with CI by
  the steps below.
---

# Test vllm37 with CI

1. Pick the workflow: `test-all.yml` to check a change, or one test: `test-concurrent-calls.yml`, `test-arrival.yml`, `test-request-length.yml`, `test-die-count.yml` or `test-weight-dtype.yml`.
2. `gh workflow run <file> -R dogkeeper886/vllm37 --ref <branch>`, `main` or the branch under test.
3. To change the values for this run, add the inputs:
   - one test: `-f calls=1,2,4`, `-f stagger=0,20`, `-f prompt=<names>`, `-f dies=1,2,4` or `-f dtype=float32,half`.
   - `test-all.yml`: `-f tests=<names>`, `-f build=true`, `-f server=local`, `-f model=<id>`, `-f max_model_len=<n>`, `-f dtype=half`, `-f dies=2`, `-f request_length_prompt=<names>`, `-f concurrent_calls=1,2,4`, `-f arrival_calls=2`, `-f arrival_stagger=0,20`, `-f die_count=1,2` or `-f weight_dtype=float32,half`. `server=local` serves the host build's working tree, uncommitted edits included; `build=true` builds the image from the run's commit first.
4. To change the standing values, set the repository variable: `gh variable list -R dogkeeper886/vllm37` shows `TEST_*`, and `gh variable set <name> -R dogkeeper886/vllm37 --body <value>` changes one.
5. `gh run list -R dogkeeper886/vllm37 --workflow <file> --limit 1`
6. `gh run watch <run-id> -R dogkeeper886/vllm37 --exit-status`
7. `gh run download <run-id> -R dogkeeper886/vllm37`, then read `report.json` and `server.json`; a `test-all.yml` run has one folder per test job and the summary table on the run page. Each row's `dies` holds every die's clock and peak temperature; a tok/s with `dies.valid: false`, struck through in the tables, measured a hot or throttled die, so quote it only with its `dies.reason`.
8. Dispatch the next test after this run ends: the lock keeps one waiting run and cancels an older one.
