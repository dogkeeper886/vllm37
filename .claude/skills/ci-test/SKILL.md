---
description: >-
  CI tests vllm37 on the self-hosted K80 runner with manual workflows on
  main: k80-test.yml, and the five test-*.yml files that call it with their
  own input. A run stops the vllm37 and ollama services, serves vllm37, sends
  short-answer and the timed requests, starts the services again and has a
  judge read every reply. An agent wants a test result and curls a server it
  started by hand. The hand test skips the judge, so its pass says nothing
  about whether the replies were language. This skill asks the agent to test
  vllm37 with CI by the steps below.
---

# Test vllm37 with CI

1. Pick the workflow: `test-concurrent-calls.yml`, `test-arrival.yml`, `test-request-length.yml`, `test-die-count.yml`, `test-weight-dtype.yml`, or `k80-test.yml` for any input.
2. `gh workflow run <file> -R dogkeeper886/vllm37 --ref main`
3. For `k80-test.yml`, add inputs: `-f server=docker|local -f dies=<n> -f dtype=float32|half -f prompt=<names> -f calls=<n,n> -f stagger=<n,n>`.
4. `gh run list -R dogkeeper886/vllm37 --workflow <file> --limit 1`
5. `gh run watch <run-id> -R dogkeeper886/vllm37 --exit-status`
6. `gh run download <run-id> -R dogkeeper886/vllm37`, then read `report.json` and `server.json`.
7. Dispatch the next test after this run ends: the lock keeps one waiting run and cancels an older one.
