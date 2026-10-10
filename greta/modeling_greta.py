from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from transformers.cache_utils import (
    Cache,
    DynamicCache,
    LinearAttentionCacheLayerMixin,
    LinearAttentionLayer,
)
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.utils import logging

from .configuration_greta import GretaConfig

try:
    from .dynamic_conv import (
        dynamic_causal_conv1d as _TRITON_DYNAMIC_CAUSAL_CONV1D,
        dynamic_causal_conv1d_decode as _TRITON_DYNAMIC_CAUSAL_CONV1D_DECODE,
        dynamic_causal_conv1d_inference as _TRITON_DYNAMIC_CAUSAL_CONV1D_INFERENCE,
    )
except ImportError:
    _TRITON_DYNAMIC_CAUSAL_CONV1D = None
    _TRITON_DYNAMIC_CAUSAL_CONV1D_DECODE = None
    _TRITON_DYNAMIC_CAUSAL_CONV1D_INFERENCE = None

logger = logging.get_logger(__name__)


class GretaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        hidden_states = hidden_states * torch.rsqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.eps)
        return (hidden_states * self.weight.float()).to(input_dtype)


class GretaRotaryEmbedding(nn.Module):
    def __init__(self, config: GretaConfig, theta: float):
        super().__init__()
        rotary_dim = config.rotary_dim
        inv_freq = 1.0 / (
            theta
            ** (
                torch.arange(0, rotary_dim, 2, dtype=torch.float32)
                / float(rotary_dim)
            )
        )

        self.register_buffer("inv_freq", inv_freq, persistent=True)

    @torch.no_grad()
    def forward(
        self, hidden_states: torch.Tensor, position_ids: torch.LongTensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq.to(device=hidden_states.device, dtype=torch.float32)
        freqs = position_ids.to(torch.float32).unsqueeze(-1) * inv_freq.view(1, 1, -1)
        embeddings = torch.cat((freqs, freqs), dim=-1)
        return embeddings.cos().to(hidden_states.dtype), embeddings.sin().to(hidden_states.dtype)


def rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    first, second = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rotary_pos_emb(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    rotary_dim = cos.shape[-1]

    q_rot, q_pass = query_states[..., :rotary_dim], query_states[..., rotary_dim:]
    k_rot, k_pass = key_states[..., :rotary_dim], key_states[..., rotary_dim:]
    query_states = torch.cat(((q_rot * cos) + (rotate_half(q_rot) * sin), q_pass), dim=-1)
    key_states = torch.cat(((k_rot * cos) + (rotate_half(k_rot) * sin), k_pass), dim=-1)
    return query_states, key_states


def repeat_kv(hidden_states: torch.Tensor, num_repeats: int) -> torch.Tensor:
    if num_repeats == 1:
        return hidden_states
    return hidden_states.repeat_interleave(num_repeats, dim=1)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Any,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attention_weights = torch.matmul(
        query.float(), key_states.float().transpose(-2, -1)
    ) * scaling
    if attention_mask is not None:
        if isinstance(attention_mask, torch.Tensor) and attention_mask.dtype == torch.bool:
            attention_weights = attention_weights.masked_fill(
                ~attention_mask, torch.finfo(attention_weights.dtype).min
            )
        else:
            attention_weights = attention_weights + attention_mask.to(attention_weights.dtype)

    attention_weights = F.softmax(attention_weights, dim=-1, dtype=torch.float32)
    attention_weights = F.dropout(
        attention_weights, p=dropout, training=module.training
    )
    attention_output = torch.matmul(attention_weights, value_states.float()).to(query.dtype)
    attention_output = attention_output.transpose(1, 2).contiguous()
    return attention_output, attention_weights.to(query.dtype)


def _resolve_attention_dtype(config: GretaConfig, hidden_states: torch.Tensor) -> torch.dtype:
    requested = config.attention_compute_dtype
    if requested == "model" or hidden_states.device.type != "cuda":
        return hidden_states.dtype
    mapping = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    return mapping[requested]


def _stable_group_rms_norm(hidden_states: torch.Tensor, eps: float) -> torch.Tensor:
    """RMS-normalize the final group axis without squaring large values directly."""

    max_abs = hidden_states.abs().amax(dim=-1, keepdim=True)
    safe_scale = max_abs.clamp_min(1.0)
    scaled = hidden_states / safe_scale
    denom = safe_scale * torch.sqrt(
        scaled.pow(2).mean(dim=-1, keepdim=True) + eps / safe_scale.pow(2)
    )
    return hidden_states / denom


class GretaConvCacheLayer(LinearAttentionLayer):
    def batch_repeat_interleave(self, repeats: int) -> None:
        for state_idx in range(self.number_of_states):
            if self.is_conv_states_initialized[state_idx]:
                self.conv_states[state_idx] = self.conv_states[state_idx].repeat_interleave(repeats, dim=0)
            if self.is_recurrent_states_initialized[state_idx]:
                self.recurrent_states[state_idx] = self.recurrent_states[state_idx].repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        for state_idx in range(self.number_of_states):
            if self.is_conv_states_initialized[state_idx]:
                device_indices = indices.to(self.conv_states[state_idx].device)
                self.conv_states[state_idx] = self.conv_states[state_idx].index_select(0, device_indices)
            if self.is_recurrent_states_initialized[state_idx]:
                device_indices = indices.to(self.recurrent_states[state_idx].device)
                self.recurrent_states[state_idx] = self.recurrent_states[state_idx].index_select(0, device_indices)


class GretaDynamicCache(DynamicCache):
    def __init__(
        self,
        ddp_cache_data=None,
        config: GretaConfig | None = None,
        offloading: bool = False,
        offload_only_non_sliding: bool = False,
    ):
        super().__init__(
            ddp_cache_data=ddp_cache_data,
            config=config,
            offloading=offloading,
            offload_only_non_sliding=offload_only_non_sliding,
        )
        if config is not None and ddp_cache_data is None:
            num_states = getattr(config, "number_of_conv_states", 1)
            for layer_idx, layer_type in enumerate(config.layer_types):
                if layer_type == "linear_attention":
                    self.layers[layer_idx] = GretaConvCacheLayer(number_of_states=num_states)

    def get_conv_state_for_decode(
        self,
        layer_idx: int,
        state_idx: int = 0,
    ) -> torch.Tensor | None:
        layer = self.layers[layer_idx]
        if (
            not isinstance(layer, GretaConvCacheLayer)
            or layer.record_past
            or not self.has_previous_state(layer_idx=layer_idx, state_idx=state_idx)
            or not layer.is_conv_states_initialized[state_idx]
        ):
            return None
        return layer.conv_states[state_idx]

    def update_conv_inputs(
        self,
        conv_inputs: torch.Tensor,
        layer_idx: int,
        kernel_size: int,
        state_idx: int = 0,
    ) -> torch.Tensor:
        history_size = kernel_size - 1
        if history_size == 0:
            return conv_inputs

        had_previous_state = self.has_previous_state(layer_idx=layer_idx, state_idx=state_idx)
        cache_layer = self.layers[layer_idx]

        if not had_previous_state and not cache_layer.record_past:
            context = F.pad(conv_inputs, (0, 0, history_size, 0))
            state_inputs = conv_inputs[:, -history_size:, :].transpose(1, 2).contiguous()
            self.update_conv_state(
                state_inputs,
                layer_idx=layer_idx,
                state_idx=state_idx,
                conv_kernel_size=history_size,
            )
            return context

        channel_first = conv_inputs.transpose(1, 2).contiguous()
        full_states = self.update_conv_state(
            channel_first,
            layer_idx=layer_idx,
            state_idx=state_idx,
            conv_kernel_size=history_size,
        )
        if had_previous_state:
            return full_states.transpose(1, 2).contiguous()
        return F.pad(conv_inputs, (0, 0, history_size, 0))


def _generic_update_conv_inputs(
    cache: Cache,
    conv_inputs: torch.Tensor,
    layer_idx: int,
    kernel_size: int,
) -> torch.Tensor:
    if isinstance(cache, GretaDynamicCache):
        return cache.update_conv_inputs(conv_inputs, layer_idx, kernel_size)

    history_size = kernel_size - 1
    if history_size == 0:
        return conv_inputs
    if not hasattr(cache, "update_conv_state"):
        raise TypeError(
            f"{type(cache).__name__} cannot store Greta dynamic-convolution state. "
            "Use GretaDynamicCache or a Transformers Cache with update_conv_state()."
        )

    layer = cache.layers[layer_idx]
    if not isinstance(layer, LinearAttentionCacheLayerMixin):
        raise TypeError(
            f"Cache layer {layer_idx} is {type(layer).__name__}, but Greta expects a convolution-state cache layer."
        )
    had_previous_state = cache.has_previous_state(layer_idx=layer_idx, state_idx=0)
    channel_first = conv_inputs.transpose(1, 2).contiguous()
    full_states = cache.update_conv_state(
        channel_first,
        layer_idx=layer_idx,
        state_idx=0,
        conv_kernel_size=history_size,
    )
    context = full_states if had_previous_state else F.pad(channel_first, (history_size, 0))

    return context.transpose(1, 2).contiguous()


def torch_dynamic_causal_conv1d_decode(
    x: torch.Tensor,
    weights: torch.Tensor,
    gate: torch.Tensor | None,
    conv_state: torch.Tensor,
    group_size: int,
    norm: bool,
    eps: float,
) -> torch.Tensor:
    batch_size, hidden_size = x.shape
    kernel_size = weights.shape[1]
    num_filters = hidden_size // group_size
    history_size = kernel_size - 1

    grouped_state = conv_state.float().view(
        batch_size, num_filters, group_size, history_size
    )
    grouped_x = x.float().view(batch_size, num_filters, group_size)
    compact_weights = weights.float().permute(0, 2, 1)
    output = (
        grouped_state * compact_weights[..., :-1].unsqueeze(2)
    ).sum(dim=-1)
    output = output + grouped_x * compact_weights[..., -1].unsqueeze(-1)
    if norm:
        output = _stable_group_rms_norm(output, eps)
    output = output.reshape(batch_size, hidden_size).to(x.dtype)
    if gate is not None:
        output = output * F.silu(gate)

    if history_size == 1:
        conv_state[..., 0].copy_(x)
    else:
        next_state = torch.cat((conv_state[..., 1:], x.unsqueeze(-1)), dim=-1)
        conv_state.copy_(next_state)
    return output


def torch_dynamic_causal_conv1d_from_context(
    context: torch.Tensor,
    weights: torch.Tensor,
    gate: torch.Tensor | None,
    group_size: int,
    norm: bool,
    eps: float,
) -> torch.Tensor:
    batch_size, query_length, kernel_size, num_filters = weights.shape
    hidden_size = context.shape[-1]
    if hidden_size != num_filters * group_size:
        raise ValueError(
            f"hidden_size={hidden_size} must equal num_filters*group_size="
            f"{num_filters}*{group_size}"
        )
    if context.shape[1] != query_length + kernel_size - 1:
        raise ValueError(
            "Dynamic-convolution context must contain exactly W-1 history tokens plus the current query chunk."
        )
        
    windows = context.unfold(dimension=1, size=kernel_size, step=1)
    windows = windows.reshape(batch_size, query_length, num_filters, group_size, kernel_size)
    compact_weights = weights.permute(0, 1, 3, 2).unsqueeze(3)
    output = (windows.float() * compact_weights.float()).sum(dim=-1)

    if norm:
        output = _stable_group_rms_norm(output, eps)

    output = output.reshape(batch_size, query_length, hidden_size).to(context.dtype)
    if gate is not None:
        output = output * F.silu(gate)
    return output


def triton_dynamic_causal_conv1d_from_context(
    context: torch.Tensor,
    weights: torch.Tensor,
    gate: torch.Tensor | None,
    group_size: int,
    norm: bool,
    eps: float,
) -> torch.Tensor:
    if _TRITON_DYNAMIC_CAUSAL_CONV1D_INFERENCE is None:
        raise RuntimeError("The Triton dynamic-convolution inference kernel is unavailable")
    return _TRITON_DYNAMIC_CAUSAL_CONV1D_INFERENCE(
        context=context,
        weight=weights,
        gate=gate,
        group_size=group_size,
        norm=norm,
        norm_eps=eps,
    )


def triton_dynamic_causal_conv1d_autograd_from_context(
    context: torch.Tensor,
    weights: torch.Tensor,
    gate: torch.Tensor | None,
    group_size: int,
    norm: bool,
    eps: float,
) -> torch.Tensor:
    """Compatibility path for autograd with non-packed HF batches.

    Efficient training should call the packed kernel directly with `cu_seqlens`.
    """
    if _TRITON_DYNAMIC_CAUSAL_CONV1D is None:
        raise RuntimeError("The Triton dynamic-convolution training kernel is unavailable")

    batch_size, query_length, kernel_size, num_filters = weights.shape
    context_length = context.shape[1]
    outputs = []
    for batch_idx in range(batch_size):
        padded_weights = weights.new_zeros((context_length, kernel_size, num_filters))
        padded_weights[-query_length:] = weights[batch_idx]
        padded_gate = None
        if gate is not None:
            padded_gate = gate.new_zeros((1, context_length, gate.shape[-1]))
            padded_gate[:, -query_length:] = gate[batch_idx : batch_idx + 1]
        full_output = _TRITON_DYNAMIC_CAUSAL_CONV1D(
            context[batch_idx : batch_idx + 1].contiguous(),
            padded_weights.contiguous(),
            gate=padded_gate.contiguous() if padded_gate is not None else None,
            cu_seqlens=None,
            group_size=group_size,
            norm=norm,
            norm_eps=eps,
            cu_seqlens_cpu=None,
        )
        outputs.append(full_output[:, -query_length:])
    return torch.cat(outputs, dim=0)


class GretaDynamicConv(nn.Module):
    def __init__(self, config: GretaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.kernel_size = config.kernel_size
        self.group_size = config.conv_group_size
        self.num_filters = config.num_conv_filters
        self.norm = config.dynconv_norm
        self.norm_eps = config.dynconv_norm_eps

        self.kernel_proj = nn.Linear(
            config.hidden_dim,
            self.num_filters * self.kernel_size,
            bias=True,
        )
        self.gate_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.in_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, query_length, _ = hidden_states.shape
        if attention_mask is not None:
            if attention_mask.ndim != 2 or attention_mask.shape != (batch_size, query_length):
                raise ValueError("Convolution attention_mask must match [batch, query_length]")
            # Mask every layer, not just the embedding: attention and the MLP
            # can populate masked positions again. Zero inputs before projection
            # so padding contributes neither convolution history nor gates.
            hidden_states = hidden_states.masked_fill(~attention_mask.bool().unsqueeze(-1), 0)
        weights = self.kernel_proj(hidden_states).view(
            batch_size,
            query_length,
            self.kernel_size,
            self.num_filters,
        )
        conv_inputs = self.in_proj(hidden_states)
        gate = self.gate_proj(hidden_states)

        needs_grad = torch.is_grad_enabled() and (
            conv_inputs.requires_grad
            or weights.requires_grad
            or gate.requires_grad
        )

        decode_state = None
        if (
            use_cache
            and query_length == 1
            and not needs_grad
            and self.kernel_size > 1
            and isinstance(past_key_values, GretaDynamicCache)
        ):
            decode_state = past_key_values.get_conv_state_for_decode(self.layer_idx)
            if decode_state is not None and decode_state.device != conv_inputs.device:
                decode_state = None

        if decode_state is not None:
            current_input = conv_inputs[:, 0, :]
            current_weights = weights[:, 0, :, :].to(current_input.dtype)
            current_gate = gate[:, 0, :]
            if (
                self.config.use_triton_kernels
                and _TRITON_DYNAMIC_CAUSAL_CONV1D_DECODE is not None
                and current_input.is_cuda
                and decode_state.is_contiguous()
            ):
                output = _TRITON_DYNAMIC_CAUSAL_CONV1D_DECODE(
                    x=current_input,
                    weight=current_weights,
                    conv_state=decode_state,
                    gate=current_gate,
                    group_size=self.group_size,
                    norm=self.norm,
                    norm_eps=self.norm_eps,
                )
            else:
                output = torch_dynamic_causal_conv1d_decode(
                    x=current_input,
                    weights=current_weights,
                    gate=current_gate,
                    conv_state=decode_state,
                    group_size=self.group_size,
                    norm=self.norm,
                    eps=self.norm_eps,
                )
            return self.out_proj(output.unsqueeze(1))

        if use_cache and past_key_values is not None:
            context = _generic_update_conv_inputs(
                past_key_values,
                conv_inputs,
                self.layer_idx,
                self.kernel_size,
            )
        else:
            context = F.pad(conv_inputs, (0, 0, self.kernel_size - 1, 0))

        needs_grad = torch.is_grad_enabled() and (
            context.requires_grad
            or weights.requires_grad
            or gate.requires_grad
        )
        use_triton_inference = (
            self.config.use_triton_kernels
            and _TRITON_DYNAMIC_CAUSAL_CONV1D_INFERENCE is not None
            and context.is_cuda
            and not needs_grad
        )
        use_triton_autograd = (
            self.config.use_triton_kernels
            and _TRITON_DYNAMIC_CAUSAL_CONV1D is not None
            and context.is_cuda
            and needs_grad
        )

        if use_triton_inference:
            output = triton_dynamic_causal_conv1d_from_context(
                context=context,
                weights=weights.to(context.dtype),
                gate=gate,
                group_size=self.group_size,
                norm=self.norm,
                eps=self.norm_eps,
            )
        elif use_triton_autograd:
            output = triton_dynamic_causal_conv1d_autograd_from_context(
                context=context,
                weights=weights.to(context.dtype),
                gate=gate,
                group_size=self.group_size,
                norm=self.norm,
                eps=self.norm_eps,
            )
        else:
            if self.config.use_triton_kernels and context.is_cuda:
                logger.warning_once(
                    "Greta's Triton dynamic-convolution kernels are unavailable; using the PyTorch fallback."
                )
            output = torch_dynamic_causal_conv1d_from_context(
                context=context,
                weights=weights,
                gate=gate,
                group_size=self.group_size,
                norm=self.norm,
                eps=self.norm_eps,
            )
        return self.out_proj(output)


class GretaAttention(nn.Module):
    def __init__(self, config: GretaConfig, layer_idx: int, layer_type: str):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = layer_type
        self.num_heads = config.num_q_heads
        self.num_key_value_heads = config.num_kv_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.qk_dim = config.qk_dim
        self.v_dim = config.v_dim
        self.scaling = self.qk_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.sliding_window = (
            config.sliding_window if layer_type == "sliding_attention" else None
        )

        self.q_proj = nn.Linear(config.hidden_dim, self.qk_dim * self.num_heads, bias=False)
        self.k_proj = nn.Linear(
            config.hidden_dim, self.qk_dim * self.num_key_value_heads, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_dim, self.v_dim * self.num_key_value_heads, bias=False
        )
        self.g_proj = nn.Linear(config.hidden_dim, self.num_heads, bias=False)
        self.q_layernorm = GretaRMSNorm(self.qk_dim, eps=config.norm_eps)
        self.k_layernorm = GretaRMSNorm(self.qk_dim, eps=config.norm_eps)
        self.out_proj = nn.Linear(self.v_dim * self.num_heads, config.hidden_dim, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Any = None,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, query_length, _ = hidden_states.shape
        query_states = self.q_layernorm(
            self.q_proj(hidden_states).view(
                batch_size, query_length, self.num_heads, self.qk_dim
            )
        )
        key_states = self.k_layernorm(
            self.k_proj(hidden_states).view(
                batch_size, query_length, self.num_key_value_heads, self.qk_dim
            )
        )
        value_states = self.v_proj(hidden_states).view(
            batch_size, query_length, self.num_key_value_heads, self.v_dim
        )
        gate = torch.sigmoid(self.g_proj(hidden_states)).unsqueeze(-1)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        query_states = query_states.transpose(1, 2).contiguous()
        key_states = key_states.transpose(1, 2).contiguous()
        value_states = value_states.transpose(1, 2).contiguous()

        compute_dtype = _resolve_attention_dtype(self.config, hidden_states)
        query_states = query_states.to(compute_dtype)
        key_states = key_states.to(compute_dtype)
        value_states = value_states.to(compute_dtype)

        if use_cache and past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attention_output, attention_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attention_output = attention_output.to(hidden_states.dtype)
        attention_output = attention_output * gate.to(hidden_states.dtype)
        attention_output = attention_output.reshape(batch_size, query_length, -1).contiguous()
        return self.out_proj(attention_output), attention_weights


class GretaMLP(nn.Module):
    def __init__(self, config: GretaConfig):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_dim, config.ffn_dim, bias=False)
        self.w3 = nn.Linear(config.hidden_dim, config.ffn_dim, bias=False)
        self.w2 = nn.Linear(config.ffn_dim, config.hidden_dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(hidden_states)) * self.w3(hidden_states))


class GretaDecoderLayer(nn.Module):
    def __init__(self, config: GretaConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        self.operator_norm = GretaRMSNorm(config.hidden_dim, eps=config.norm_eps)
        self.ffn_norm = GretaRMSNorm(config.hidden_dim, eps=config.norm_eps)
        self.feed_forward = GretaMLP(config)
        if self.layer_type == "linear_attention":
            self.conv = GretaDynamicConv(config, layer_idx)
        else:
            self.self_attn = GretaAttention(config, layer_idx, self.layer_type)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Any,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        past_key_values: Cache | None,
        use_cache: bool,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        residual = hidden_states
        normalized = self.operator_norm(hidden_states)
        if self.layer_type == "linear_attention":
            mixed = self.conv(
                normalized, past_key_values=past_key_values, use_cache=use_cache,
                attention_mask=attention_mask,
            )
            attention_weights = None
        else:
            mixed, attention_weights = self.self_attn(
                normalized,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )
        hidden_states = residual + mixed
        hidden_states = hidden_states + self.feed_forward(self.ffn_norm(hidden_states))
        return hidden_states, attention_weights


class GretaPreTrainedModel(PreTrainedModel):
    config_class = GretaConfig
    base_model_prefix = "model"
    _no_split_modules = ["GretaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, GretaRMSNorm):
            module.weight.data.fill_(1.0)


class GretaModel(GretaPreTrainedModel):
    def __init__(self, config: GretaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.embedding_dim, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [GretaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GretaRMSNorm(config.hidden_dim, eps=config.norm_eps)
        self.rotary_emb = GretaRotaryEmbedding(config, theta=config.rope_theta)
        self.short_rotary_emb = GretaRotaryEmbedding(config, theta=config.short_rope_theta)
        self.first_full_attention_layer_idx = next(
            (i for i, layer_type in enumerate(config.layer_types) if layer_type == "full_attention"),
            None,
        )
        self.first_sliding_attention_layer_idx = next(
            (
                i
                for i, layer_type in enumerate(config.layer_types)
                if layer_type == "sliding_attention"
            ),
            None,
        )
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | dict[str, Any] | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> BaseModelOutputWithPast | tuple:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        use_cache = self.config.use_cache if use_cache is None else use_cache
        output_attentions = (
            self.config.output_attentions if output_attentions is None else output_attentions
        )
        output_hidden_states = (
            self.config.output_hidden_states
            if output_hidden_states is None
            else output_hidden_states
        )
        return_dict = self.config.return_dict if return_dict is None else return_dict

        if output_attentions and self.config._attn_implementation != "eager":
            raise ValueError(
                "output_attentions=True requires attn_implementation='eager'. "
                "Optimized attention backends do not return a full attention matrix."
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        batch_size, query_length, _ = inputs_embeds.shape

        if use_cache and past_key_values is None:
            past_key_values = GretaDynamicCache(config=self.config)
        if past_key_values is not None and not isinstance(past_key_values, Cache):
            raise TypeError(
                "Transformers 5 no longer accepts tuple past_key_values. Pass a Cache instance, "
                "preferably GretaDynamicCache(config=model.config)."
            )

        active_past_key_values = past_key_values if use_cache else None

        # Generation may supply precomputed 4D attention masks. Retain the
        # original 2D padding mask alongside them for convolution and positions.
        padding_mask = (
            attention_mask.get("linear_attention")
            if isinstance(attention_mask, dict)
            else attention_mask
            if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2
            else None
        )

        if position_ids is None:
            if padding_mask is not None:
                position_ids = padding_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(padding_mask == 0, 0)
                position_ids = position_ids[:, -query_length:]
            else:
                past_seen_tokens = (
                    active_past_key_values.get_seq_length()
                    if active_past_key_values is not None
                    else 0
                )
                position_ids = torch.arange(
                    past_seen_tokens,
                    past_seen_tokens + query_length,
                    device=inputs_embeds.device,
                    dtype=torch.long,
                ).unsqueeze(0)
                position_ids = position_ids.expand(batch_size, -1)

        if isinstance(attention_mask, dict):
            causal_mask_mapping = attention_mask
        else:
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": active_past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping: dict[str, Any] = {}
            if self.first_full_attention_layer_idx is not None:
                causal_mask_mapping["full_attention"] = create_causal_mask(
                    **mask_kwargs, layer_idx=self.first_full_attention_layer_idx
                )
            if self.first_sliding_attention_layer_idx is not None:
                causal_mask_mapping["sliding_attention"] = (
                    create_sliding_window_causal_mask(
                        **mask_kwargs, layer_idx=self.first_sliding_attention_layer_idx
                    )
                )

        position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        if self.first_full_attention_layer_idx is not None:
            position_embeddings["full_attention"] = self.rotary_emb(
                inputs_embeds, position_ids
            )
        if self.first_sliding_attention_layer_idx is not None:
            position_embeddings["sliding_attention"] = self.short_rotary_emb(
                inputs_embeds, position_ids
            )

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if decoder_layer.layer_type == "linear_attention":
                layer_attention_mask = padding_mask[:, -query_length:] if padding_mask is not None else None
                layer_position_embeddings = None
            else:
                if decoder_layer.layer_type not in causal_mask_mapping:
                    raise ValueError(
                        f"Missing attention mask for layer type {decoder_layer.layer_type!r}"
                    )
                layer_attention_mask = causal_mask_mapping[decoder_layer.layer_type]
                layer_position_embeddings = position_embeddings[decoder_layer.layer_type]

            hidden_states, attention_weights = decoder_layer(
                hidden_states,
                attention_mask=layer_attention_mask,
                position_embeddings=layer_position_embeddings,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )
            if output_attentions:
                all_attentions += (attention_weights,)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            values = (
                hidden_states,
                past_key_values if use_cache else None,
                all_hidden_states,
                all_attentions,
            )
            return tuple(value for value in values if value is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class GretaForCausalLM(GretaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    def __init__(self, config: GretaConfig):
        super().__init__(config)
        self.model = GretaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.lm_head = new_embeddings

    def embeddings_are_tied(self) -> bool:
        """Return whether the LM head and token embedding share one parameter."""

        input_weight = self.get_input_embeddings().weight
        output_weight = self.get_output_embeddings().weight
        if input_weight is output_weight:
            return True
        if input_weight.device.type == "meta" or output_weight.device.type == "meta":
            return False
        try:
            return (
                input_weight.untyped_storage().data_ptr()
                == output_weight.untyped_storage().data_ptr()
                and input_weight.storage_offset() == output_weight.storage_offset()
                and input_weight.shape == output_weight.shape
                and input_weight.stride() == output_weight.stride()
            )
        except RuntimeError:
            return False

    def _set_embeddings_tied(self, tied: bool) -> None:
        """Synchronize the config and the actual embedding parameter alias."""

        self.config.tie_word_embeddings = tied
        self.config.tie_embeddings = tied
        if tied:
            self.lm_head.weight = self.model.embed_tokens.weight
        elif self.embeddings_are_tied():
            self.lm_head.weight = nn.Parameter(self.lm_head.weight.detach().clone())

    def set_decoder(self, decoder: GretaModel) -> None:
        self.model = decoder

    def get_decoder(self) -> GretaModel:
        return self.model

    @staticmethod
    def create_masks_for_generate(
        config: GretaConfig,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None,
        position_ids: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        config = config.get_text_config()
        layer_types = config.layer_types
        first_full_layer = next(
            (i for i, layer_type in enumerate(layer_types) if layer_type == "full_attention"),
            None,
        )
        first_sliding_layer = next(
            (
                i
                for i, layer_type in enumerate(layer_types)
                if layer_type == "sliding_attention"
            ),
            None,
        )

        mask_kwargs = {
            "config": config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
            "or_mask_function": kwargs.get("or_mask_function"),
            "and_mask_function": kwargs.get("and_mask_function"),
            "block_sequence_ids": kwargs.get("block_sequence_ids"),
        }
        masks: dict[str, Any] = {}
        if attention_mask is not None and attention_mask.ndim == 2:
            masks["linear_attention"] = attention_mask
        if first_full_layer is not None:
            masks["full_attention"] = create_causal_mask(
                **mask_kwargs, layer_idx=first_full_layer
            )
        if first_sliding_layer is not None:
            masks["sliding_attention"] = create_sliding_window_causal_mask(
                **mask_kwargs, layer_idx=first_sliding_layer
            )
        return masks

    def _prepare_cache_for_generation(
        self,
        generation_config,
        model_kwargs: dict,
        generation_mode,
        batch_size: int,
        max_cache_length: int,
    ):
        # Transformers 5.17 no longer checks the old _supports_*_cache flags.
        # Static caches also assume identical Q/K and V dimensions, which Greta
        # does not require, and quantized caches cannot store its conv history.
        if generation_config.use_cache is not False and generation_config.cache_implementation not in (
            None, "dynamic", "offloaded"
        ):
            raise ValueError(
                "Greta supports cache_implementation='dynamic' or 'offloaded' only."
            )
        user_supplied = model_kwargs.get("past_key_values") is not None
        result = GenerationMixin._prepare_cache_for_generation(
            self,
            generation_config,
            model_kwargs,
            generation_mode,
            batch_size,
            max_cache_length,
        )
        if (
            not user_supplied
            and generation_config.use_cache is not False
            and generation_config.cache_implementation in (None, "dynamic", "offloaded")
            and type(model_kwargs.get("past_key_values")) is DynamicCache
        ):
            cache = GretaDynamicCache(
                config=self.config,
                offloading=generation_config.cache_implementation == "offloaded",
                offload_only_non_sliding=False,
            )
            if generation_config.is_assistant:
                cache.activate_past_recording()
            model_kwargs["past_key_values"] = cache
        return result

    @staticmethod
    def convert_legacy_state_dict(state_dict: dict[str, torch.Tensor]) -> OrderedDict:
        converted = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[len("module.") :]
            if key.startswith(("embed_tokens.", "layers.", "norm.")):
                key = f"model.{key}"
            converted[key] = value
        return converted

    def load_legacy_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        strict: bool = True,
        assign: bool = False,
        tie_embeddings: bool | None = None,
    ):
        converted = self.convert_legacy_state_dict(state_dict)
        should_tie = (
            bool(self.config.tie_word_embeddings)
            if tie_embeddings is None
            else bool(tie_embeddings)
        )

        embedding_key = "model.embed_tokens.weight"
        lm_head_key = "lm_head.weight"
        embedding_weight = converted.get(embedding_key)
        lm_head_weight = converted.get(lm_head_key)

        converted.setdefault(
            "model.rotary_emb.inv_freq", self.model.rotary_emb.inv_freq.detach().clone()
        )
        converted.setdefault(
            "model.short_rotary_emb.inv_freq",
            self.model.short_rotary_emb.inv_freq.detach().clone(),
        )

        if should_tie:
            if embedding_weight is not None and lm_head_weight is not None:
                if embedding_weight.shape != lm_head_weight.shape or not torch.equal(
                    embedding_weight.detach().cpu(), lm_head_weight.detach().cpu()
                ):
                    raise ValueError(
                        "tie_embeddings=True, but embed_tokens.weight and "
                        "lm_head.weight in the checkpoint are not identical."
                    )
            elif embedding_weight is not None:
                converted[lm_head_key] = embedding_weight
            elif lm_head_weight is not None:
                converted[embedding_key] = lm_head_weight

        self._set_embeddings_tied(should_tie)
        result = self.load_state_dict(converted, strict=strict, assign=assign)
        self._set_embeddings_tied(should_tie)
        return result

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | dict[str, Any] | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast | tuple:
        return_dict = self.config.return_dict if return_dict is None else return_dict
        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )

        hidden_states = model_outputs.last_hidden_state
        if isinstance(logits_to_keep, int) and logits_to_keep < 0:
            raise ValueError("logits_to_keep must be non-negative")
        if labels is not None:
            selected_hidden_states = hidden_states
        elif isinstance(logits_to_keep, int):
            selected_hidden_states = hidden_states if logits_to_keep == 0 else hidden_states[:, -logits_to_keep:, :]
        else:
            selected_hidden_states = hidden_states[:, logits_to_keep, :]

        logits = self.lm_head(selected_hidden_states).float()
        if self.config.logit_soft_cap is not None:
            cap = float(self.config.logit_soft_cap)
            logits = cap * torch.tanh(logits / cap)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        if not return_dict:
            output = (
                logits,
                model_outputs.past_key_values,
                model_outputs.hidden_states,
                model_outputs.attentions,
            )
            output = tuple(value for value in output if value is not None)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=model_outputs.past_key_values,
            hidden_states=model_outputs.hidden_states,
            attentions=model_outputs.attentions,
        )


GretaModel.register_for_auto_class("AutoModel")
GretaForCausalLM.register_for_auto_class("AutoModelForCausalLM")


__all__ = [
    "GretaConfig",
    "GretaDynamicCache",
    "GretaModel",
    "GretaForCausalLM",
]
