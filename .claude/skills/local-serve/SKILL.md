---
description: >-
  The host build of vllm37 serves as the user service vllm37, which reads its
  model, context length and GPU from ~/.config/vllm37/serve.env. An agent
  wants a server and starts serve.sh or python -m vllm in the background. The
  loose process holds a GPU and port 8000 where systemctl and CI cannot see
  it, so the next CI run fails on the port and nothing puts the server back.
  This skill asks the agent to serve the host build by the steps below.
---

# Serve the host build as the user service vllm37

1. `tools/k80-host/install-service.sh`
2. Set `PROFILE` to a profile in `docker/k80/models/` and `CUDA_VISIBLE_DEVICES` to its dies in `~/.config/vllm37/serve.env`, with `MODEL`, `DTYPE`, `TP_SIZE` and `MAX_MODEL_LEN` commented out.
3. `systemctl --user restart vllm37`
4. Repeat `curl -fsS localhost:8000/v1/models` until it answers.
5. If the service stops instead, read `~/.local/state/vllm37/serve.log`.
