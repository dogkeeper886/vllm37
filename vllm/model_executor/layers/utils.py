# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility methods for model layers."""
from typing import Callable, Optional

import torch

from vllm import _custom_ops as ops
from vllm import envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils import direct_register_custom_op

logger = init_logger(__name__)


def shuffle_weight(w: torch.Tensor) -> torch.Tensor:
    # Shuffle weight along the last dimension so that
    # we folded the weights to adjance location
    # Example:
    # input:
    #       [[1, 2, 3, 4, 5, 6],
    #        [7, 8, 9, 10, 11, 12]]
    # output:
    #       [[1, 4, 2, 5, 3, 6],
    #        [7, 10, 8, 11, 9, 12]]
    # This will be used together with triton swiglu kernel
    shape = w.shape
    N = shape[-1]
    first = w[..., :N // 2]
    second = w[..., N // 2:]

    stacked = torch.stack((first, second), dim=-1)
    w_shuffled = stacked.reshape(shape)
    return w_shuffled


def get_token_bin_counts_and_mask(
    tokens: torch.Tensor,
    vocab_size: int,
    num_seqs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Compute the bin counts for the tokens.
    # vocab_size + 1 for padding.
    bin_counts = torch.zeros((num_seqs, vocab_size + 1),
                             dtype=torch.long,
                             device=tokens.device)
    bin_counts.scatter_add_(1, tokens, torch.ones_like(tokens))
    bin_counts = bin_counts[:, :vocab_size]
    mask = bin_counts > 0

    return bin_counts, mask


def apply_penalties(logits: torch.Tensor, prompt_tokens_tensor: torch.Tensor,
                    output_tokens_tensor: torch.Tensor,
                    presence_penalties: torch.Tensor,
                    frequency_penalties: torch.Tensor,
                    repetition_penalties: torch.Tensor) -> torch.Tensor:
    """
    Applies penalties in place to the logits tensor
    logits : The input logits tensor of shape [num_seqs, vocab_size]
    prompt_tokens_tensor: A tensor containing the prompt tokens. The prompts 
        are padded to the maximum prompt length within the batch using 
        `vocab_size` as the padding value. The value `vocab_size` is used 
        for padding because it does not correspond to any valid token ID 
        in the vocabulary.
    output_tokens_tensor: The output tokens tensor.
    presence_penalties: The presence penalties of shape (num_seqs, )
    frequency_penalties: The frequency penalties of shape (num_seqs, )
    repetition_penalties: The repetition penalties of shape (num_seqs, )
    """
    num_seqs, vocab_size = logits.shape
    _, prompt_mask = get_token_bin_counts_and_mask(prompt_tokens_tensor,
                                                   vocab_size, num_seqs)
    output_bin_counts, output_mask = get_token_bin_counts_and_mask(
        output_tokens_tensor, vocab_size, num_seqs)

    # Apply repetition penalties as a custom op
    from vllm._custom_ops import apply_repetition_penalties
    apply_repetition_penalties(logits, prompt_mask, output_mask,
                               repetition_penalties)

    # We follow the definition in OpenAI API.
    # Refer to https://platform.openai.com/docs/api-reference/parameter-details
    logits -= frequency_penalties.unsqueeze(dim=1) * output_bin_counts
    logits -= presence_penalties.unsqueeze(dim=1) * output_mask
    return logits


# K80 fork (#95). On Kepler, cuBLAS runs a product of 2 or more rows on a
# kernel that costs several times its one-row kernel: on a K80 die the
# Qwen3-0.6B lm_head takes 3.7 ms for one row and 18.2 ms for two. Two other
# ways avoid it for a small batch:
#   bias   F.linear with a bias goes through cuBLASLt, which picks a good
#          kernel in FP32 up to about 6 rows (lm_head, 4 rows: 3.9 ms). A
#          layer without a bias gets a zero one. No effect in FP16.
#   rows   a batched product against the same weight (batch stride 0) runs
#          the one-row kernel for each row in one launch, and gives each row
#          bit for bit what it gets alone. The faster way in FP16.
# Which is fastest depends on the shape, so the first product of each shape,
# dtype and row count times the ways and keeps the fastest. One row, and more
# than 8 rows (prefill), always call F.linear.
_SMALL_BATCH_MAX_ROWS = 8
_small_batch_way: dict[tuple, str] = {}
_zero_bias: dict[tuple, torch.Tensor] = {}
_is_kepler: Optional[bool] = None


def _by_rows(x: torch.Tensor, weight: torch.Tensor,
             bias: Optional[torch.Tensor]) -> torch.Tensor:
    m = x.shape[0]
    w = weight.t().unsqueeze(0).expand(m, -1, -1)
    if bias is None:
        return torch.bmm(x.unsqueeze(1), w).squeeze(1)
    return torch.baddbmm(bias.expand(m, 1, -1), x.unsqueeze(1), w).squeeze(1)


def _ms(fn: Callable[[], torch.Tensor], iters: int = 5) -> float:
    fn()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def _with_bias(x: torch.Tensor, weight: torch.Tensor,
               bias: Optional[torch.Tensor]) -> torch.Tensor:
    if bias is None:
        key = (weight.shape[0], weight.dtype, weight.device)
        bias = _zero_bias.get(key)
        if bias is None:
            bias = _zero_bias[key] = weight.new_zeros(weight.shape[0])
    return torch.nn.functional.linear(x, weight, bias)


def _small_batch_gemm(x: torch.Tensor, weight: torch.Tensor,
                      bias: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """The product the fastest way for a small batch, or None for F.linear."""
    global _is_kepler
    if not envs.VLLM_K80_SMALL_BATCH_GEMM or not x.is_cuda:
        return None
    if _is_kepler is None:
        _is_kepler = torch.cuda.get_device_capability(x.device)[0] < 5
    m = x.numel() // x.shape[-1] if x.shape[-1] else 0
    if (not _is_kepler or not 2 <= m <= _SMALL_BATCH_MAX_ROWS
            or weight.dim() != 2 or x.dtype != weight.dtype
            or torch.compiler.is_compiling()):
        return None
    x2 = x.reshape(m, x.shape[-1])
    key = (weight.shape[0], weight.shape[1], weight.dtype, m, bias is None)
    way = _small_batch_way.get(key)
    if way is None:
        # A CUDA graph capture cannot time the ways; vLLM runs each batch
        # size uncaptured first, so the choice is normally already made.
        if torch.cuda.is_current_stream_capturing():
            return None
        ways = {
            "linear": lambda: torch.nn.functional.linear(x2, weight, bias),
            "rows": lambda: _by_rows(x2, weight, bias),
        }
        if bias is None and weight.dtype == torch.float32:
            ways["bias"] = lambda: _with_bias(x2, weight, None)
        times = {name: _ms(fn) for name, fn in ways.items()}
        way = _small_batch_way[key] = min(times, key=times.__getitem__)
        logger.debug("K80 small-batch GEMM %s: %s (%s)", key, way,
                     ", ".join(f"{k} {v:.3f} ms" for k, v in times.items()))
    if way == "linear":
        return None
    out = (_by_rows(x2, weight, bias)
           if way == "rows" else _with_bias(x2, weight, bias))
    return out.view(*x.shape[:-1], weight.shape[0])


def default_unquantized_gemm(layer: torch.nn.Module,
                             x: torch.Tensor,
                             weight: torch.Tensor,
                             bias: Optional[torch.Tensor] = None):
    out = _small_batch_gemm(x, weight, bias)
    if out is not None:
        return out
    return torch.nn.functional.linear(x, weight, bias)


def rocm_unquantized_gemm_impl(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    from vllm.platforms.rocm import on_gfx9
    k = weight.shape[1]
    use_skinny = (envs.VLLM_ROCM_USE_SKINNY_GEMM and on_gfx9() and \
                    x.dtype in [torch.float16, torch.bfloat16] \
                    and k % 8 == 0 and bias is None)

    if use_skinny is not True:
        return torch.nn.functional.linear(x, weight, bias)

    x_view = x.view(-1, x.size(-1))
    n = x_view.shape[0]
    m = weight.shape[0]
    cu_count = current_platform.get_cu_count()

    if m > 8 and 0 < n <= 4:
        out = ops.wvSplitK(weight, x_view, cu_count)
        return out.view(*x.shape[:-1], weight.shape[0])
    elif m % 4 == 0 and n == 1 and k <= 8192:
        out = ops.LLMM1(weight, x_view, 4)
        return out.view(*x.shape[:-1], weight.shape[0])
    return torch.nn.functional.linear(x, weight, bias)


def rocm_unquantized_gemm_impl_fake(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


def rocm_unquantized_gemm(layer: torch.nn.Module,
                          x: torch.Tensor,
                          weight: torch.Tensor,
                          bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.ops.vllm.rocm_unquantized_gemm_impl(x, weight, bias)


direct_register_custom_op(
    op_name="rocm_unquantized_gemm_impl",
    op_func=rocm_unquantized_gemm_impl,
    mutates_args=[],
    fake_impl=rocm_unquantized_gemm_impl_fake,
    dispatch_key=current_platform.dispatch_key,
)


def cpu_unquantized_gemm(layer: torch.nn.Module,
                         x: torch.Tensor,
                         weight: torch.Tensor,
                         bias: Optional[torch.Tensor] = None):
    if getattr(layer, "use_cpu_sgl", False):
        return torch.ops._C.weight_packed_linear(x, weight, bias, True)
    else:
        return torch.nn.functional.linear(x, weight, bias)


def dispatch_unquantized_gemm() -> Callable[..., torch.Tensor]:
    if current_platform.is_rocm():
        return rocm_unquantized_gemm
    elif current_platform.is_cpu():
        return cpu_unquantized_gemm
    else:
        return default_unquantized_gemm
