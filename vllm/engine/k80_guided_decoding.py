# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# K80 fork (#86): guided decoding (guided_json, guided_regex, guided_choice,
# guided_grammar, structural tags) on the V0 engine. Upstream removed V0's
# guided decoding (37efc63b64, "[V0 deprecation] Guided decoding", #21347)
# and kept it in V1 only; the fork runs V0, so a request's
# SamplingParams.guided_decoding was accepted and ignored, and the reply came
# back unconstrained with no error.
#
# This builds one V0 logits processor per sequence from V1's xgrammar
# backend: the same validation, grammar compiler and matcher. Each step it
# feeds the matcher the tokens generated since the last step, fills the
# allowed-token bitmask on the CPU and masks the logits on the GPU with bit
# operations (xgrammar's own mask kernel needs Triton, which Kepler lacks).

from typing import Optional

import torch

from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer import AnyTokenizer

logger = init_logger(__name__)

# Request and engine backends this path serves.
_BACKENDS = (None, "", "auto", "xgrammar")


class GuidedLogitsProcessor:
    """Masks one sequence's logits to the tokens its grammar allows."""

    def __init__(self, grammar, vocab_size: int):
        import xgrammar as xgr
        self.grammar = grammar  # a V1 XgrammarGrammar
        self.vocab_size = vocab_size
        self.bitmask = xgr.allocate_token_bitmask(1, vocab_size)
        self._shifts: Optional[torch.Tensor] = None

    def clone(self) -> "GuidedLogitsProcessor":
        """A fresh matcher on the same compiled grammar (parallel sampling
        clones the processor per sequence)."""
        import xgrammar as xgr

        from vllm.v1.structured_output.backend_xgrammar import XgrammarGrammar
        ctx = self.grammar.ctx
        return GuidedLogitsProcessor(
            XgrammarGrammar(vocab_size=self.vocab_size,
                            matcher=xgr.GrammarMatcher(ctx),
                            ctx=ctx), self.vocab_size)

    def __call__(self, past_token_ids: list[int],
                 logits: torch.Tensor) -> torch.Tensor:
        new = past_token_ids[self.grammar.num_processed_tokens:]
        if new and not self.grammar.accept_tokens("v0", list(new)):
            return logits
        if self.grammar.is_terminated():
            return logits
        self.grammar.fill_bitmask(self.bitmask, 0)
        if self._shifts is None or self._shifts.device != logits.device:
            self._shifts = torch.arange(32,
                                        dtype=torch.int32,
                                        device=logits.device)
        words = self.bitmask[0].to(logits.device, non_blocking=True)
        allowed = ((words.unsqueeze(-1) >> self._shifts) & 1).flatten().bool()
        n = min(allowed.numel(), logits.shape[-1])
        logits[:n].masked_fill_(~allowed[:n], float("-inf"))
        if logits.shape[-1] > n:  # padded vocabulary rows
            logits[n:] = float("-inf")
        return logits


class K80GuidedDecoding:
    """Per-engine xgrammar backends (one per tokenizer)."""

    def __init__(self, vllm_config, vocab_size: int):
        self.vllm_config = vllm_config
        self.vocab_size = vocab_size
        self._backends: dict[int, object] = {}

    def _backend(self, tokenizer: AnyTokenizer):
        backend = self._backends.get(id(tokenizer))
        if backend is None:
            from vllm.v1.structured_output.backend_xgrammar import (
                XgrammarBackend)
            backend = XgrammarBackend(self.vllm_config,
                                      tokenizer=tokenizer,
                                      vocab_size=self.vocab_size)
            self._backends[id(tokenizer)] = backend
        return backend

    def logits_processor(self, params: SamplingParams,
                         tokenizer: AnyTokenizer) -> GuidedLogitsProcessor:
        """The processor for params.guided_decoding; ValueError when the
        request cannot be served."""
        from vllm.v1.structured_output.backend_xgrammar import (
            validate_xgrammar_grammar)
        from vllm.v1.structured_output.request import (
            get_structured_output_key)
        guided = params.guided_decoding
        engine_backend = self.vllm_config.decoding_config.backend
        if (guided.backend not in _BACKENDS
                or not (engine_backend in _BACKENDS
                        or engine_backend.startswith("xgrammar"))):
            raise ValueError(
                "Guided decoding on the K80 (V0 engine) supports the xgrammar "
                f"backend only, not {guided.backend or engine_backend!r}")
        validate_xgrammar_grammar(params)
        request_type, spec = get_structured_output_key(params)
        grammar = self._backend(tokenizer).compile_grammar(request_type, spec)
        return GuidedLogitsProcessor(grammar, self.vocab_size)
