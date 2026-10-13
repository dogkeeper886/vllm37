---
description: >-
  CI builds the vllm37-local image on the self-hosted K80 runner from a branch
  on GitHub, with the manual workflow k80-build.yml. An agent runs gh from
  this checkout, which has the fork and upstream vLLM as remotes. gh resolves
  the repository to upstream, or the run builds a branch that lacks the local
  commits, and the agent reports a build that never ran on the fork's code.
  This skill asks the agent to build the image with CI by the steps below.
---

# Build the vllm37-local image with CI

1. `git push -u origin <branch>`
2. `gh workflow run k80-build.yml -R dogkeeper886/vllm37 --ref <branch>`
3. `gh run list -R dogkeeper886/vllm37 --workflow k80-build.yml --branch <branch> --limit 1`
4. `gh run watch <run-id> -R dogkeeper886/vllm37 --exit-status`
