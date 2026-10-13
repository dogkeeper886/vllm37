---
description: >-
  vllm37 runs on the K80 host from an editable install in ~/.venvs/vllm37,
  built by tools/k80-host/setup.sh against CUDA 11.4 and the sm_37 PyTorch. An
  agent sees a Python package and runs pip install on it. The plain install
  replaces the sm_37 PyTorch with a CUDA 12 build the K80 cannot run, and the
  vllm37 service keeps serving the old build. This skill asks the agent to
  build vllm37 on the host by the steps below.
---

# Build vllm37 on the host filesystem

1. `systemctl --user stop vllm37`
2. `tools/k80-host/setup.sh`
3. `systemctl --user start vllm37`
