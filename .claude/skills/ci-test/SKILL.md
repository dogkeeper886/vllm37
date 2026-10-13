---
description: >-
  CI tests vllm37 on the self-hosted K80 runner with the manual workflow
  k80-runtime.yml: it stops the vllm37 and ollama services, serves the
  vllm37 container, compares /v1/completions with the golden output and
  starts the services again. An agent wants a test result and curls a server
  it started by hand. The hand test skips the kernel test, the backend check
  and the golden comparison, so its pass says nothing about what CI would
  report. This skill asks the agent to test vllm37 with CI by the steps
  below.
---

# Test vllm37 with CI

1. `git push -u origin <branch>`
2. `gh workflow run k80-runtime.yml -R dogkeeper886/vllm37 --ref <branch>`
3. `gh run list -R dogkeeper886/vllm37 --workflow k80-runtime.yml --branch <branch> --limit 1`
4. `gh run watch <run-id> -R dogkeeper886/vllm37 --exit-status`
