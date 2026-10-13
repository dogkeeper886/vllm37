---
description: >-
  The vllm37-local image is built from this checkout on top of the
  vllm37-builder image, which holds CUDA 11.4, GCC 10 and the sm_37 PyTorch.
  An agent sees a Dockerfile and runs docker build on it. The bare build
  misses the builder image and the commit label, so it fails, or it yields an
  image CI rebuilds because the label does not match the branch. This skill
  asks the agent to build the image by the steps below.
---

# Build the vllm37-local image on this machine

1. `docker/k80/ci/ensure-builder.sh`
2. `make -C docker/k80 build-local`
