# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# set some common config/environment variables that should be set
# for all processes created by vllm and all processes
# that interact with vllm workers.
# they are executed whenever `import vllm` is called.

if os.environ.get('NCCL_CUMEM_ENABLE', '0') != '0':
    logger.warning(
        "NCCL_CUMEM_ENABLE is set to %s, skipping override. "
        "This may increase memory overhead with cudagraph+allreduce: "
        "https://github.com/NVIDIA/nccl/issues/1234",
        os.environ['NCCL_CUMEM_ENABLE'])
elif not os.path.exists('/dev/nvidia-caps-imex-channels'):
    # NCCL requires NCCL_CUMEM_ENABLE to work with
    # multi-node NVLink, typically on GB200-NVL72 systems.
    # The ultimate way to detect multi-node NVLink is to use
    # NVML APIs, which are too expensive to call here.
    # As an approximation, we check the existence of
    # /dev/nvidia-caps-imex-channels, used by
    # multi-node NVLink to communicate across nodes.
    # This will still cost some GPU memory, but it is worthwhile
    # because we can get very fast cross-node bandwidth with NVLink.
    os.environ['NCCL_CUMEM_ENABLE'] = '0'

# see https://github.com/vllm-project/vllm/pull/15951
# it avoids unintentional cuda initialization from torch.cuda.is_available()
os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '1'

# see https://github.com/vllm-project/vllm/issues/10480
os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
# see https://github.com/vllm-project/vllm/issues/10619
try:
    import torch._inductor.config as _inductor_config
    _inductor_config.compile_threads = 1
except ImportError:  # e.g. PyTorch 2.0 builds without a usable inductor
    pass

# PyTorch < 2.1 lacks float8 dtypes. Add stubs so that third-party libs
# (e.g. compressed_tensors) that reference torch.float8_e4m3fn at class
# definition time do not crash on import.
if not hasattr(torch, 'float8_e4m3fn'):
    import struct

    class _FakeFinfo:
        """Minimal finfo stub for float8 types on PyTorch < 2.1."""
        def __init__(self, *, bits, max_val, min_val, eps, tiny):
            self.bits = bits
            self.max = max_val
            self.min = min_val
            self.eps = eps
            self.tiny = tiny
            self.dtype = None  # placeholder

    # Create placeholder dtype objects (they are just sentinels, never
    # used for actual tensor creation on legacy CUDA).
    class _Float8Dtype:
        """Sentinel dtype for float8 types unavailable in this PyTorch."""
        def __init__(self, name):
            self._name = name
        def __repr__(self):
            return self._name
        def __hash__(self):
            return hash(self._name)
        def __eq__(self, other):
            return isinstance(other, _Float8Dtype) and self._name == other._name

    torch.float8_e4m3fn = _Float8Dtype('torch.float8_e4m3fn')
    torch.float8_e5m2 = _Float8Dtype('torch.float8_e5m2')
    # Also provide float8_e4m3fnuz and float8_e5m2fnuz if needed
    if not hasattr(torch, 'float8_e4m3fnuz'):
        torch.float8_e4m3fnuz = _Float8Dtype('torch.float8_e4m3fnuz')
    if not hasattr(torch, 'float8_e5m2fnuz'):
        torch.float8_e5m2fnuz = _Float8Dtype('torch.float8_e5m2fnuz')

    # Monkey-patch torch.finfo to handle our fake dtypes
    _original_finfo = torch.finfo
    _fake_finfo_map = {
        torch.float8_e4m3fn: _FakeFinfo(
            bits=8, max_val=448.0, min_val=-448.0, eps=0.125, tiny=0.015625),
        torch.float8_e5m2: _FakeFinfo(
            bits=8, max_val=57344.0, min_val=-57344.0, eps=0.25, tiny=0.0000152587890625),
        torch.float8_e4m3fnuz: _FakeFinfo(
            bits=8, max_val=240.0, min_val=-240.0, eps=0.125, tiny=0.015625),
        torch.float8_e5m2fnuz: _FakeFinfo(
            bits=8, max_val=57344.0, min_val=-57344.0, eps=0.25, tiny=0.0000152587890625),
    }

    class _PatchedFinfo:
        """Wrapper around torch.finfo that also handles fake float8 dtypes."""
        def __new__(cls, dtype):
            if dtype in _fake_finfo_map:
                return _fake_finfo_map[dtype]
            return _original_finfo(dtype)

    torch.finfo = _PatchedFinfo
