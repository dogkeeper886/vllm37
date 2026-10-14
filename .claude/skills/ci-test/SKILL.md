---
description: >-
  CI tests vllm37 on the self-hosted K80 runner with five manual workflows on
  main, test-*.yml, whose steps are the test and test-end actions. A run
  stops the vllm37 and ollama services, serves vllm37, sends short-answer and
  the timed requests, starts the services again and has a judge read every
  reply. An agent wants a test result and curls a server it started by hand.
  The hand test skips the judge, so its pass says nothing about whether the
  replies were language. This skill asks the agent to test vllm37 with CI by
  the steps below.
---

# Test vllm37 with CI

1. Pick the workflow: `test-concurrent-calls.yml`, `test-arrival.yml`, `test-request-length.yml`, `test-die-count.yml` or `test-weight-dtype.yml`.
2. `gh workflow run <file> -R dogkeeper886/vllm37 --ref main`
3. To change the values for this run, add the test's input: `-f calls=1,2,4`, `-f stagger=0,20`, `-f prompt=<names>`, `-f dies=1,2,4` or `-f dtype=float32,half`.
4. To change the standing values, set the repository variable: `gh variable list -R dogkeeper886/vllm37` shows `TEST_*`, and `gh variable set <name> -R dogkeeper886/vllm37 --body <value>` changes one.
5. `gh run list -R dogkeeper886/vllm37 --workflow <file> --limit 1`
6. `gh run watch <run-id> -R dogkeeper886/vllm37 --exit-status`
7. `gh run download <run-id> -R dogkeeper886/vllm37`, then read `report.json` and `server.json`.
8. Dispatch the next test after this run ends: the lock keeps one waiting run and cancels an older one.
