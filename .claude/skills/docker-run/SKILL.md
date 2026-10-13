---
description: >-
  The vllm37 container serves the vllm37-local image on port 8000, the port
  the vllm37 user service also uses. An agent wants the container and runs
  docker run or docker compose up by hand. The hand start skips the fixed
  container name and the readiness wait, and it collides with the service on
  the port, so the agent tests a server that is still loading, or the wrong
  server. This skill asks the agent to run the container by the steps below.
---

# Run the vllm37 container on this machine

1. `systemctl --user stop vllm37`
2. `docker/k80/container-up.sh`
3. `curl -fsS localhost:8000/v1/models`
4. `docker/k80/container-down.sh`
