# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Radix attention."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Optional

import torch
from torch import nn

from sglang.srt.compilation.compilation_config import register_split_op
from sglang.srt.compilation.piecewise_context_manager import get_forward_context
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import QuantizationConfig
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class AttentionType(Enum):
    """
    Attention type.
    Use string to be compatible with `torch.compile`.
    """

    # Decoder attention between previous layer Q/K/V
    DECODER = "decoder"
    # Decoder bidirectional attention between image tokens
    DECODER_BIDIRECTIONAL = "decoder_bidirectional"
    # Encoder attention between previous layer Q/K/V
    ENCODER_ONLY = "encoder_only"


class RadixAttention(nn.Module):
    """
    The attention layer implementation.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scaling: float,
        num_kv_heads: int,
        layer_id: int,
        logit_cap: float = 0.0,
        v_head_dim: int = -1,
        sliding_window_size: int = -1,
        is_cross_attention: bool = False,
        pos_encoding_mode: str = "NONE",
        logit_capping_method: str = "tanh",
        quant_config: Optional[QuantizationConfig] = None,
        attn_type: AttentionType = AttentionType.DECODER,
        use_irope: bool = False,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_q_head_num = num_heads
        self.tp_k_head_num = num_kv_heads
        self.tp_v_head_num = num_kv_heads
        self.head_dim = head_dim
        self.qk_head_dim = head_dim
        self.v_head_dim = v_head_dim if v_head_dim != -1 else head_dim
        self.scaling = scaling
        self.layer_id = layer_id
        self.logit_cap = logit_cap
        self.sliding_window_size = sliding_window_size or -1
        self.is_cross_attention = is_cross_attention
        self.use_irope = use_irope
        self.k_scale = None
        self.v_scale = None
        self.k_scale_float = None
        self.v_scale_float = None
        self.quant_method = None

        if quant_config is not None:
            self.quant_method = quant_config.get_quant_method(self, prefix=prefix)
        if self.quant_method is not None:
            self.quant_method.create_weights(self)
        elif (
            self.k_scale_float is None
            and self.v_scale_float is None
            and get_global_server_args().kv_cache_dtype != "auto"
        ):
            # The user opted into a quantized KV cache (e.g. fp8_e5m2) but
            # the loaded checkpoint provides no per-layer K/V scales, so no
            # BaseKVCacheMethod is attached. Several attention backends
            # (flashattention/fa3, flashinfer, trtllm) gate their fp8
            # read/dequant path on `layer.k_scale is not None` and silently
            # fall back to bf16 reads otherwise, defeating the whole point
            # of the cache. Default to unit scale here so those backends
            # actually exercise their fp8 path.
            self._set_default_kv_scales()
        self.attn_type = attn_type

        self.pos_encoding_mode = pos_encoding_mode
        self.logit_capping_method = logit_capping_method
        self.xai_temperature_len = -1

    def _set_default_kv_scales(self) -> None:
        """Default the K/V dequant scales to 1.0.

        This mirrors the no-checkpoint-scales path of
        ``BaseKVCacheMethod.process_weights_after_loading`` and is safe for
        ``fp8_e5m2``, whose 5-bit exponent already covers the bf16 dynamic
        range -- values just round-trip through fp8 without rescaling.
        ``fp8_e4m3`` would benefit from per-layer calibrated scales for
        accuracy, but that path is gated by hardware support (sm_89+) and
        is orthogonal to enabling the fp8 read path in the first place.
        """
        self.k_scale = torch.nn.Parameter(
            torch.tensor(1.0, dtype=torch.float32), requires_grad=False
        )
        self.v_scale = torch.nn.Parameter(
            torch.tensor(1.0, dtype=torch.float32), requires_grad=False
        )
        self.k_scale_float = 1.0
        self.v_scale_float = 1.0

    def forward(
        self,
        q,
        k,
        v,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        if k is not None:
            # For cross-layer sharing, kv can be None
            assert v is not None
            if "k_rope" not in kwargs:
                k = k.view(-1, self.tp_k_head_num, self.qk_head_dim)
                v = v.view(-1, self.tp_v_head_num, self.v_head_dim)
            else:
                k = k.view(-1, self.tp_k_head_num, self.v_head_dim)

        if forward_batch.forward_mode.is_extend() and get_forward_context() is not None:
            if self.qk_head_dim != self.v_head_dim:
                output = q.new_empty((q.shape[0], self.tp_q_head_num * self.v_head_dim))
            else:
                output = torch.empty_like(q)
            unified_attention_with_output(
                q, k, v, output, save_kv_cache, self.layer_id, **kwargs
            )
            return output
        else:
            return forward_batch.attn_backend.forward(
                q,
                k,
                v,
                self,
                forward_batch,
                save_kv_cache,
                **kwargs,
            )


@register_custom_op(mutates_args=["output"])
@register_split_op()
def unified_attention_with_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    save_kv_cache: bool,
    layer_id: int,
    *,
    q_rope: Optional[torch.Tensor] = None,
    k_rope: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
) -> None:
    context = get_forward_context()
    forward_batch = context.forward_batch
    attention_layers = context.attention_layers
    attention_layer = attention_layers[layer_id]

    kwargs = {}
    if q_rope is not None:
        kwargs["q_rope"] = q_rope
    if k_rope is not None:
        kwargs["k_rope"] = k_rope
    if sinks is not None:
        kwargs["sinks"] = sinks

    ret = forward_batch.attn_backend.forward(
        query, key, value, attention_layer, forward_batch, save_kv_cache, **kwargs
    )
    assert (
        output.numel() == ret.numel()
    ), f"Output tensor element mismatch: {output.numel()} != {ret.numel()}"

    output.view(ret.shape).copy_(ret)
    return
