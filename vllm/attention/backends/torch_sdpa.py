# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer using PyTorch's scaled_dot_product_attention (SDPA)
and PagedAttention for decode. Designed for legacy CUDA GPUs (sm < 60,
e.g. Tesla K80 / Kepler) that lack xformers and flash-attention support."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.nn.functional as F

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer,
                                              AttentionMetadata, AttentionType)
from vllm.attention.backends.utils import (
    CommonAttentionState, CommonMetadataBuilder,
    get_num_prefill_decode_query_kv_tokens, get_seq_len_block_table_args,
    is_all_cross_attn_metadata_set, is_all_encoder_attn_metadata_set)
from vllm.attention.ops.paged_attn import (PagedAttention,
                                           PagedAttentionMetadata)
from vllm.logger import init_logger

logger = init_logger(__name__)


# K80 fork (#96): attention scores (heads x query rows x keys) per SDPA call
# in prefill, 256 MiB in FP32; longer prompts go in chunks of query rows.
K80_SDPA_MAX_SCORES = 1 << 26


class TorchSDPABackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "TORCH_SDPA"

    @staticmethod
    def get_impl_cls() -> Type["TorchSDPAImpl"]:
        return TorchSDPAImpl

    @staticmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        return TorchSDPAMetadata

    @staticmethod
    def get_builder_cls() -> Type["TorchSDPAMetadataBuilder"]:
        return TorchSDPAMetadataBuilder

    @staticmethod
    def get_state_cls() -> Type["CommonAttentionState"]:
        return CommonAttentionState

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return PagedAttention.get_kv_cache_shape(num_blocks, block_size,
                                                 num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: Dict[int, int],
    ) -> None:
        PagedAttention.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        PagedAttention.copy_blocks(kv_caches, src_to_dists)


@dataclass
class TorchSDPAMetadata(AttentionMetadata, PagedAttentionMetadata):
    """Metadata for TorchSDPA backend.

    Reuses the same structure as XFormersMetadata — both backends use
    PagedAttention for decode and differ only in the prefill kernel.
    """

    seq_lens_tensor: Optional[torch.Tensor]
    max_prefill_seq_len: int
    max_decode_seq_len: int
    use_cuda_graph: bool

    seq_lens: Optional[List[int]] = None
    seq_start_loc: Optional[torch.Tensor] = None
    context_lens_tensor: Optional[torch.Tensor] = None
    max_query_len: Optional[int] = None
    max_decode_query_len: Optional[int] = None
    query_start_loc: Optional[torch.Tensor] = None

    _cached_prefill_metadata: Optional["TorchSDPAMetadata"] = None
    _cached_decode_metadata: Optional["TorchSDPAMetadata"] = None

    # Encoder / cross-attention fields
    encoder_seq_lens: Optional[List[int]] = None
    encoder_seq_lens_tensor: Optional[torch.Tensor] = None
    encoder_seq_start_loc: Optional[torch.Tensor] = None
    max_encoder_seq_len: Optional[int] = None
    num_encoder_tokens: Optional[int] = None
    cross_slot_mapping: Optional[torch.Tensor] = None
    cross_block_tables: Optional[torch.Tensor] = None

    @property
    def is_all_encoder_attn_metadata_set(self):
        return is_all_encoder_attn_metadata_set(self)

    @property
    def is_all_cross_attn_metadata_set(self):
        return is_all_cross_attn_metadata_set(self)

    @property
    def prefill_metadata(self) -> Optional["TorchSDPAMetadata"]:
        if self.num_prefills == 0:
            return None

        if self._cached_prefill_metadata is not None:
            return self._cached_prefill_metadata

        assert ((self.seq_lens is not None)
                or (self.encoder_seq_lens is not None))
        assert ((self.seq_lens_tensor is not None)
                or (self.encoder_seq_lens_tensor is not None))

        query_start_loc = (None if self.query_start_loc is None else
                           self.query_start_loc[:self.num_prefills + 1])
        seq_start_loc = (None if self.seq_start_loc is None else
                         self.seq_start_loc[:self.num_prefills + 1])
        slot_mapping = (None if self.slot_mapping is None else
                        self.slot_mapping[:self.num_prefill_tokens])
        seq_lens = (None if self.seq_lens is None else
                    self.seq_lens[:self.num_prefills])
        seq_lens_tensor = (None if self.seq_lens_tensor is None else
                           self.seq_lens_tensor[:self.num_prefills])
        context_lens_tensor = (None if self.context_lens_tensor is None else
                               self.context_lens_tensor[:self.num_prefills])
        block_tables = (None if self.block_tables is None else
                        self.block_tables[:self.num_prefills])

        self._cached_prefill_metadata = TorchSDPAMetadata(
            num_prefills=self.num_prefills,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=0,
            slot_mapping=slot_mapping,
            multi_modal_placeholder_index_maps=self.
            multi_modal_placeholder_index_maps,
            enable_kv_scales_calculation=self.enable_kv_scales_calculation,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            max_query_len=self.max_query_len,
            max_prefill_seq_len=self.max_prefill_seq_len,
            max_decode_seq_len=0,
            query_start_loc=query_start_loc,
            seq_start_loc=seq_start_loc,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables,
            use_cuda_graph=False,
            encoder_seq_lens=self.encoder_seq_lens,
            encoder_seq_lens_tensor=self.encoder_seq_lens_tensor,
            max_encoder_seq_len=self.max_encoder_seq_len,
            cross_slot_mapping=self.cross_slot_mapping,
            cross_block_tables=self.cross_block_tables)
        return self._cached_prefill_metadata

    @property
    def decode_metadata(self) -> Optional["TorchSDPAMetadata"]:
        if self.num_decode_tokens == 0:
            return None

        if self._cached_decode_metadata is not None:
            return self._cached_decode_metadata

        assert ((self.seq_lens_tensor is not None)
                or (self.encoder_seq_lens_tensor is not None))

        slot_mapping = (None if self.slot_mapping is None else
                        self.slot_mapping[self.num_prefill_tokens:])
        seq_lens_tensor = (None if self.seq_lens_tensor is None else
                           self.seq_lens_tensor[self.num_prefills:])
        block_tables = (None if self.block_tables is None else
                        self.block_tables[self.num_prefills:])

        self._cached_decode_metadata = TorchSDPAMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=self.num_decode_tokens,
            slot_mapping=slot_mapping,
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=True,
            seq_lens_tensor=seq_lens_tensor,
            max_prefill_seq_len=0,
            max_decode_seq_len=self.max_decode_seq_len,
            block_tables=block_tables,
            use_cuda_graph=self.use_cuda_graph,
            encoder_seq_lens=self.encoder_seq_lens,
            encoder_seq_lens_tensor=self.encoder_seq_lens_tensor,
            max_encoder_seq_len=self.max_encoder_seq_len,
            cross_slot_mapping=self.cross_slot_mapping,
            cross_block_tables=self.cross_block_tables)

        if self._cached_decode_metadata.query_start_loc is not None:
            qs = self._cached_decode_metadata.query_start_loc
            self._cached_decode_metadata.query_start_loc = qs - qs[0]
        return self._cached_decode_metadata


class TorchSDPAMetadataBuilder(CommonMetadataBuilder[TorchSDPAMetadata]):
    _metadata_cls = TorchSDPAMetadata


class TorchSDPAImpl(AttentionImpl[TorchSDPAMetadata]):
    """Attention implementation using PyTorch's F.scaled_dot_product_attention
    for prefill and PagedAttention CUDA kernels for decode.

    This backend works on all CUDA GPUs including Kepler (sm_37) since it
    relies only on PyTorch built-in ops (no xformers, no flash-attn).
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: Optional[str] = None,
        use_irope: bool = False,
        sinks: Optional[torch.Tensor] = None,
    ) -> None:
        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError("KV sharing is not supported in V0 "
                                      "TORCH_SDPA backend.")
        # K80 fork (#129): attention sinks (gpt-oss), one logit per head with
        # no value. The model loads them after this, into the same tensor.
        if sinks is not None and sinks.shape != (num_heads, ):
            raise ValueError(f"Attention sinks have shape {sinks.shape}, "
                             f"not ({num_heads}, )")
        if sinks is not None and alibi_slopes is not None:
            raise NotImplementedError("TorchSDPA: attention sinks with ALiBi")
        self.sinks = sinks
        if logits_soft_cap is not None:
            logger.warning_once(
                "TorchSDPA does not support logits soft cap. "
                "Outputs may be slightly off.")
        if use_irope:
            logger.warning_once(
                "Using irope in TorchSDPA is not supported yet.")

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.attn_type = attn_type

        # Check if F.scaled_dot_product_attention supports 'scale' kwarg
        # (added in PyTorch 2.1). Cache for use in forward().
        try:
            import torch.nn.functional as _F
            _F.scaled_dot_product_attention(
                torch.empty(1, 1, 1, 1), torch.empty(1, 1, 1, 1),
                torch.empty(1, 1, 1, 1), scale=1.0)
            self._sdpa_has_scale = True
        except TypeError:
            self._sdpa_has_scale = False

        supported_head_sizes = PagedAttention.get_supported_head_sizes()
        if head_size not in supported_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by PagedAttention. "
                f"Supported head sizes are: {supported_head_sizes}.")

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        kv_cache: torch.Tensor,
        attn_metadata: "TorchSDPAMetadata",
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with Torch SDPA for prefill and PagedAttention
        for decode.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size * num_kv_heads * head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        if output_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported"
                " for TorchSDPAImpl")

        attn_type = self.attn_type

        if (attn_type == AttentionType.ENCODER
                and (not attn_metadata.is_all_encoder_attn_metadata_set)):
            raise AttributeError("Encoder attention requires setting "
                                 "encoder metadata attributes.")
        elif (attn_type == AttentionType.ENCODER_DECODER
              and (not attn_metadata.is_all_cross_attn_metadata_set)):
            raise AttributeError("Encoder/decoder cross-attention "
                                 "requires setting cross-attention "
                                 "metadata attributes.")

        query = query.view(-1, self.num_heads, self.head_size)
        if key is not None:
            assert value is not None
            key = key.view(-1, self.num_kv_heads, self.head_size)
            value = value.view(-1, self.num_kv_heads, self.head_size)
        else:
            assert value is None

        if (attn_type != AttentionType.ENCODER and kv_cache.numel() > 0):
            key_cache, value_cache = PagedAttention.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_size)

            if (key is not None) and (value is not None):
                if attn_type == AttentionType.ENCODER_DECODER:
                    updated_slot_mapping = attn_metadata.cross_slot_mapping
                else:
                    updated_slot_mapping = attn_metadata.slot_mapping

                PagedAttention.write_to_paged_cache(
                    key, value, key_cache, value_cache, updated_slot_mapping,
                    self.kv_cache_dtype, layer._k_scale, layer._v_scale)

        (num_prefill_query_tokens, num_prefill_kv_tokens,
         num_decode_query_tokens) = \
            get_num_prefill_decode_query_kv_tokens(attn_metadata, attn_type)

        output = torch.empty_like(query)
        decode_query = query[num_prefill_query_tokens:]
        query = query[:num_prefill_query_tokens]
        if key is not None and value is not None:
            key = key[:num_prefill_kv_tokens]
            value = value[:num_prefill_kv_tokens]

        assert query.shape[0] == num_prefill_query_tokens
        assert decode_query.shape[0] == num_decode_query_tokens

        if prefill_meta := attn_metadata.prefill_metadata:
            if kv_cache.numel() == 0 or prefill_meta.block_tables.numel() == 0:
                # Normal attention (no cached prefix).
                out = self._run_sdpa_forward(
                    query, key, value, prefill_meta, attn_type=attn_type)
                assert out.shape == output[:num_prefill_query_tokens].shape
                output[:num_prefill_query_tokens] = out
            else:
                assert attn_type != AttentionType.ENCODER_ONLY, (
                    "Encoder-only models should not have prefix attention.")
                assert prefill_meta.query_start_loc is not None
                assert prefill_meta.max_query_len is not None

                out = PagedAttention.forward_prefix(
                    query,
                    key,
                    value,
                    self.kv_cache_dtype,
                    key_cache,
                    value_cache,
                    prefill_meta.block_tables,
                    prefill_meta.query_start_loc,
                    prefill_meta.seq_lens_tensor,
                    prefill_meta.max_query_len,
                    self.alibi_slopes,
                    self.sliding_window,
                    layer._k_scale,
                    layer._v_scale,
                )
                assert output[:num_prefill_query_tokens].shape == out.shape
                output[:num_prefill_query_tokens] = out

        if decode_meta := attn_metadata.decode_metadata:
            assert attn_type != AttentionType.ENCODER_ONLY, (
                "Encoder-only models should not have decode metadata.")

            (
                seq_lens_arg,
                max_seq_len_arg,
                block_tables_arg,
            ) = get_seq_len_block_table_args(decode_meta, False, attn_type)

            output[num_prefill_query_tokens:] = PagedAttention.forward_decode(
                decode_query,
                key_cache,
                value_cache,
                block_tables_arg,
                seq_lens_arg,
                max_seq_len_arg,
                self.kv_cache_dtype,
                self.num_kv_heads,
                self.scale,
                self.alibi_slopes,
                layer._k_scale,
                layer._v_scale,
                sinks=None if self.sinks is None else self.sinks.float(),
                sliding_window=self.sliding_window,
            )

        return output.view(-1, self.num_heads * self.head_size)

    def _run_sdpa_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: TorchSDPAMetadata,
        attn_type: str = AttentionType.DECODER,
    ) -> torch.Tensor:
        """Run prefill attention using PyTorch's F.scaled_dot_product_attention.

        Processes each prompt individually since SDPA doesn't natively support
        variable-length sequences packed into a single tensor.

        Args:
            query: [num_prefill_tokens, num_heads, head_size]
            key: [num_prefill_tokens, num_kv_heads, head_size]
            value: [num_prefill_tokens, num_kv_heads, head_size]
            attn_metadata: attention metadata
            attn_type: type of attention
        Returns:
            [num_prefill_tokens, num_heads, head_size]
        """
        if attn_type == AttentionType.ENCODER:
            assert attn_metadata.encoder_seq_lens is not None
            seq_lens = attn_metadata.encoder_seq_lens
        elif attn_type == AttentionType.ENCODER_DECODER:
            assert attn_metadata.seq_lens is not None
            assert attn_metadata.encoder_seq_lens is not None
            q_seq_lens = attn_metadata.seq_lens
            kv_seq_lens = attn_metadata.encoder_seq_lens
        else:
            assert attn_metadata.seq_lens is not None
            seq_lens = attn_metadata.seq_lens

        is_causal = (attn_type == AttentionType.DECODER)

        output = torch.empty_like(query)
        q_start = 0
        kv_start = 0

        if attn_type == AttentionType.ENCODER_DECODER:
            iter_lens = zip(q_seq_lens, kv_seq_lens)
        else:
            iter_lens = ((s, s) for s in seq_lens)

        for q_len, kv_len in iter_lens:
            q_end = q_start + q_len
            kv_end = kv_start + kv_len

            # Extract per-prompt tensors: [seq_len, num_heads, head_size]
            q = query[q_start:q_end]
            k = key[kv_start:kv_end]
            v = value[kv_start:kv_end]

            # GQA/MQA: expand KV heads to match query heads
            if self.num_queries_per_kv > 1:
                k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
                v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

            # SDPA expects [batch, num_heads, seq_len, head_size]
            q = q.unsqueeze(0).transpose(1, 2)  # [1, num_heads, q_len, head]
            k = k.unsqueeze(0).transpose(1, 2)  # [1, num_heads, kv_len, head]
            v = v.unsqueeze(0).transpose(1, 2)  # [1, num_heads, kv_len, head]

            # K80 fork (#96): the math SDPA path that Kepler takes holds the
            # [num_heads, q_len, kv_len] scores at once, so a long prompt
            # ran out of memory and took the engine down (3.18 GiB in FP32
            # for about 7,300 tokens of Qwen3-0.6B). Query rows go in chunks
            # whose scores stay within K80_SDPA_MAX_SCORES.
            rows = max(1, K80_SDPA_MAX_SCORES // (q.shape[1] * kv_len))
            window = self.sliding_window if is_causal else None
            if q_len <= rows:
                out = self._sdpa(q, k, v, kv_len - q_len, is_causal, 0,
                                 window)
            else:
                out = torch.empty_like(q)
                for a in range(0, q_len, rows):
                    b = min(a + rows, q_len)
                    # A causal row attends to the keys up to its own position,
                    # and with a sliding window (#129) to the last window.
                    kv_end = kv_len - q_len + b if is_causal else kv_len
                    kv_lo = (max(0, kv_len - q_len + a - window + 1)
                             if window is not None else 0)
                    out[:, :, a:b] = self._sdpa(q[:, :, a:b],
                                                k[:, :, kv_lo:kv_end],
                                                v[:, :, kv_lo:kv_end],
                                                kv_len - q_len + a, is_causal,
                                                kv_lo, window)
            # out: [1, num_heads, q_len, head_size] -> [q_len, num_heads, head]
            out = out.squeeze(0).transpose(0, 1)
            output[q_start:q_end] = out

            q_start = q_end
            kv_start = kv_end

        return output

    def _sdpa(self,
              q: torch.Tensor,
              k: torch.Tensor,
              v: torch.Tensor,
              q_pos: int,
              is_causal: bool,
              k_pos: int = 0,
              window: Optional[int] = None) -> torch.Tensor:
        """F.scaled_dot_product_attention for queries [1, H, q_len, D] at
        key positions q_pos .. q_pos + q_len - 1 over keys [1, H, kv_len, D]
        from position k_pos, with ALiBi, the causal mask, the sliding window
        and attention sinks (#129)."""
        q_len, kv_len = q.shape[2], k.shape[2]
        if self.sinks is not None:
            return self._attend_with_sinks(q, k, v, q_pos, is_causal, k_pos,
                                           window)
        attn_mask = None
        if window is not None:
            key_pos = torch.arange(k_pos, k_pos + kv_len, device=q.device)
            query_pos = torch.arange(q_pos, q_pos + q_len, device=q.device)
            attn_mask = ((key_pos[None, :] <= query_pos[:, None]) &
                         (key_pos[None, :] > query_pos[:, None] - window))
            is_causal = False  # the mask holds it
        elif self.alibi_slopes is not None or (is_causal and
                                                (q_pos or q_len != kv_len)):
            key_pos = torch.arange(kv_len, device=q.device)
            query_pos = torch.arange(q_pos, q_pos + q_len, device=q.device)
            if self.alibi_slopes is not None:
                slopes = self.alibi_slopes.to(q.device, dtype=q.dtype)
                attn_mask = slopes[:, None, None] * (
                    key_pos[None, None, :] -
                    query_pos[None, :, None]).to(q.dtype)
                if is_causal:
                    attn_mask = attn_mask.masked_fill(
                        key_pos[None, None, :] > query_pos[None, :, None],
                        float("-inf"))
                attn_mask = attn_mask.unsqueeze(0)  # [1, H, q_len, kv_len]
            else:
                attn_mask = key_pos[None, :] <= query_pos[:, None]
            is_causal = False  # the mask holds it
        if self._sdpa_has_scale:
            out = F.scaled_dot_product_attention(q,
                                                 k,
                                                 v,
                                                 attn_mask=attn_mask,
                                                 scale=self.scale,
                                                 is_causal=is_causal)
        else:
            # PyTorch < 2.1: scale not supported, pre-scale query
            import math
            default_scale = 1.0 / math.sqrt(q.size(-1))
            if self.scale != default_scale:
                q = q * (self.scale / default_scale)
            out = F.scaled_dot_product_attention(q,
                                                 k,
                                                 v,
                                                 attn_mask=attn_mask,
                                                 is_causal=is_causal)
        return out

    def _attend_with_sinks(self, q: torch.Tensor, k: torch.Tensor,
                           v: torch.Tensor, q_pos: int, is_causal: bool,
                           k_pos: int, window: Optional[int]) -> torch.Tensor:
        """K80 fork (#129): attention with a sink per head, one more logit
        with no value in each softmax, computed in the query dtype as the
        math SDPA path does: the scores of this call are held once."""
        q_len, kv_len = q.shape[2], k.shape[2]
        scores = torch.matmul(q, k.transpose(-1, -2)).mul_(self.scale)
        if is_causal:
            key_pos = torch.arange(k_pos, k_pos + kv_len, device=q.device)
            query_pos = torch.arange(q_pos, q_pos + q_len, device=q.device)
            masked = key_pos[None, :] > query_pos[:, None]
            if window is not None:
                masked |= key_pos[None, :] <= query_pos[:, None] - window
            scores.masked_fill_(masked, float("-inf"))
        sinks = self.sinks.to(scores.dtype).view(1, -1, 1, 1)
        top = torch.maximum(scores.amax(-1, keepdim=True), sinks)
        probs = scores.sub_(top).exp_()
        total = probs.sum(-1, keepdim=True) + torch.exp(sinks - top)
        return torch.matmul(probs, v).div_(total)
