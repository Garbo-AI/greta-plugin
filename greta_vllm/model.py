"""Native vLLM 0.30.0 Greta implementation (single GPU).

The backbone is compiled with torch.compile and runs under CUDA graphs. The
dynamic convolution is the custom op ``vllm::greta_dynamic_conv``; it is one of
vLLM's splitting ops (registered by ``greta_vllm.register``), so mixed
prefill/decode batches run it eagerly between piecewise graphs, while
decode-only batches capture it inside a full graph.
"""
from collections.abc import Iterable
from copy import copy

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.config.compilation import CompilationMode
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFuncCalculator, MambaStateDtypeCalculator,
    MambaStateShapeCalculator, is_conv_state_dim_first,
)
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid
from vllm.sequence import IntermediateTensors
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.short_conv_attn import ShortConvAttentionMetadata

from greta.modeling_greta import (
    GretaAttention, GretaMLP, GretaRMSNorm, GretaRotaryEmbedding,
    apply_rotary_pos_emb,
)
from .conv import metadata_to_batch, packed_dynamic_conv
from .triton_conv import packed_dynamic_conv_triton, supports as triton_supports

CONV_OP = 'vllm::greta_dynamic_conv'


def validate_config(vllm_config):
    model, cache = vllm_config.model_config, vllm_config.cache_config
    parallel, config = vllm_config.parallel_config, model.hf_config
    compilation = vllm_config.compilation_config
    if (compilation.mode == CompilationMode.VLLM_COMPILE
            and compilation.cudagraph_mode.has_piecewise_cudagraphs()
            and CONV_OP not in (compilation.splitting_ops or [])):
        raise ValueError(f'{CONV_OP} is not a vLLM splitting op; load the plugin '
                         '(VLLM_PLUGINS=greta, or call greta_vllm.register() before '
                         'creating the engine) or use --enforce-eager')
    for name in ('tensor_parallel_size', 'pipeline_parallel_size',
                 'decode_context_parallel_size', 'prefill_context_parallel_size',
                 'data_parallel_size'):
        if getattr(parallel, name, 1) != 1:
            raise ValueError(f'Greta currently requires {name}=1')
    if vllm_config.quant_config is not None or model.quantization is not None:
        raise ValueError('Greta quantization is not implemented')
    if vllm_config.speculative_config is not None:
        raise ValueError('Greta speculative decoding is not implemented')
    if vllm_config.lora_config is not None:
        raise ValueError('Greta LoRA serving is not implemented')
    if cache.enable_prefix_caching or cache.mamba_cache_mode != 'none':
        raise ValueError('Greta requires enable_prefix_caching=False and mamba_cache_mode="none"')
    if cache.cache_dtype != 'auto' or cache.mamba_cache_dtype != 'auto':
        raise ValueError('Greta requires automatic KV and convolution-cache dtypes')
    if getattr(cache, 'use_replayssm', False) or getattr(cache, 'use_kda_recoverssm', False):
        raise ValueError('ReplaySSM/RecoverSSM do not implement Greta dynamic convolution')
    if config.qk_dim != config.v_dim:
        raise ValueError('This Greta adapter currently requires qk_dim == v_dim')
    if config.kernel_size < 2:
        raise ValueError('This Greta adapter requires kernel_size >= 2 for its state cache')
    if not any(t in ('full_attention', 'sliding_attention') for t in config.layer_types):
        raise ValueError('This hybrid adapter requires at least one attention layer')
    if not any(t == 'linear_attention' for t in config.layer_types):
        raise ValueError('This hybrid adapter requires at least one convolution layer')
    expected = {'bfloat16': torch.bfloat16, 'float16': torch.float16,
                'float32': torch.float32}.get(config.attention_compute_dtype, model.dtype)
    if expected != model.dtype:
        raise ValueError('Set vLLM dtype to attention_compute_dtype, or set '
                         'attention_compute_dtype="model" in the HF config')
    if model.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError('Use dtype="bfloat16" or "float16" for vLLM attention')


class GretaConv(nn.Module, MambaBase):
    """Expose dynamic-convolution history to vLLM's hybrid cache manager."""
    def __init__(self, config, vllm_config, prefix):
        super().__init__()
        self.config = config
        self.vllm_config = vllm_config
        self.prefix = prefix
        self.kernel_proj = nn.Linear(config.hidden_dim,
                                     config.num_conv_filters * config.kernel_size, bias=True)
        self.gate_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.in_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        context = vllm_config.compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f'Duplicate layer name: {prefix}')
        context[prefix] = self
        self.kv_cache = (torch.empty(0),)

    @property
    def mamba_type(self):
        # Reuse scheduler metadata and cache allocation, NOT the fixed-weight
        # short-convolution operator: Greta generates new filters at every token.
        return MambaAttentionBackendEnum.SHORT_CONV

    def get_state_shape(self):
        return GretaForCausalLM.get_mamba_state_shape_from_config(self.vllm_config)

    def get_state_dtype(self):
        return GretaForCausalLM.get_mamba_state_dtype_from_config(self.vllm_config)

    def forward(self, hidden_states, positions):
        # Projections stay in the compiled graph; only the stateful part is the op.
        n = hidden_states.shape[0]
        x = self.in_proj(hidden_states)
        gate = self.gate_proj(hidden_states)
        weights = self.kernel_proj(hidden_states).view(n, self.config.kernel_size,
                                                      self.config.num_conv_filters)
        mixed = torch.empty_like(x)
        torch.ops.vllm.greta_dynamic_conv(x, weights, gate, positions, mixed, self.prefix)
        return self.out_proj(mixed)

    def _batch(self, metadata, positions, tokens):
        """Packed-batch tensors (starts, seq_idx, slots, initial, n) for the kernels.

        Uses device tensors and shapes only, so a decode-only step is graph-capturable.
        """
        device = positions.device
        if metadata is None:
            # Profile/warmup: one request without history, written to a scratch slot.
            starts = torch.arange(2, device=device, dtype=torch.int32) * tokens
            seq_idx = torch.zeros(tokens, device=device, dtype=torch.int32)
            slots = torch.zeros(1, device=device, dtype=torch.int32)
            return starts, seq_idx, slots, torch.zeros(1, device=device, dtype=torch.int8), tokens
        if not isinstance(metadata, ShortConvAttentionMetadata):
            raise TypeError('Greta requires SHORT_CONV attention metadata')
        return metadata_to_batch(metadata, positions)

    def run_conv(self, x, weights, gate, positions, output, context: ForwardContext):
        metadata = None
        if context.attn_metadata is not None:
            metadata = context.attn_metadata[self.prefix]
        if metadata is None:
            state = x.new_empty((1, self.config.hidden_dim, self.config.kernel_size - 1))
        else:
            state = self.kv_cache[0]
            if not is_conv_state_dim_first():
                state = state.transpose(-1, -2)
        starts, seq_idx, slots, initial, n = self._batch(metadata, positions, x.shape[0])
        if n < x.shape[0]:
            output[n:].zero_()
        if n == 0:
            return
        args = (x[:n], weights[:n], gate[:n], state, starts, slots, initial,
                self.config.conv_group_size, self.config.dynconv_norm,
                self.config.dynconv_norm_eps)
        if x.is_cuda and triton_supports(self.config.conv_group_size, self.config.hidden_dim):
            packed_dynamic_conv_triton(*args, seq_idx=seq_idx, out=output[:n])
        else:
            output[:n] = packed_dynamic_conv(*args)


def greta_dynamic_conv(x: torch.Tensor, weights: torch.Tensor, gate: torch.Tensor,
                       positions: torch.Tensor, output: torch.Tensor, layer_name: str) -> None:
    context = get_forward_context()
    context.no_compile_layers[layer_name].run_conv(x, weights, gate, positions, output, context)


def greta_dynamic_conv_fake(x: torch.Tensor, weights: torch.Tensor, gate: torch.Tensor,
                            positions: torch.Tensor, output: torch.Tensor, layer_name: str) -> None:
    return


direct_register_custom_op(
    op_name='greta_dynamic_conv',
    op_func=greta_dynamic_conv,
    mutates_args=['output'],
    fake_impl=greta_dynamic_conv_fake,
)


class GretaPagedAttention(GretaAttention):
    def __init__(self, config, layer_idx, layer_type, vllm_config, prefix):
        super().__init__(config, layer_idx, layer_type)
        # Prevent the model-level sliding_window from truncating FULL layers.
        cache = copy(vllm_config.cache_config)
        cache.sliding_window = None
        self.attn = Attention(
            self.num_heads, self.qk_dim, self.scaling,
            num_kv_heads=self.num_key_value_heads,
            cache_config=cache,
            per_layer_sliding_window=self.sliding_window,
            prefix=f'{prefix}.attn',
        )

    def forward(self, hidden_states, position_embeddings):
        n = hidden_states.shape[0]
        q = self.q_layernorm(self.q_proj(hidden_states).view(1, n, self.num_heads, self.qk_dim))
        k = self.k_layernorm(self.k_proj(hidden_states).view(1, n, self.num_key_value_heads, self.qk_dim))
        v = self.v_proj(hidden_states)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        output = self.attn(q.reshape(n, -1), k.reshape(n, -1), v)
        output = output.reshape(n, self.num_heads, self.v_dim)
        output = output * torch.sigmoid(self.g_proj(hidden_states)).unsqueeze(-1)
        return self.out_proj(output.reshape(n, -1))


class GretaLayer(nn.Module):
    def __init__(self, config, index, vllm_config, prefix):
        super().__init__()
        self.layer_type = config.layer_types[index]
        self.operator_norm = GretaRMSNorm(config.hidden_dim, config.norm_eps)
        self.ffn_norm = GretaRMSNorm(config.hidden_dim, config.norm_eps)
        self.feed_forward = GretaMLP(config)
        if self.layer_type == 'linear_attention':
            self.conv = GretaConv(config, vllm_config, f'{prefix}.conv')
        else:
            self.self_attn = GretaPagedAttention(config, index, self.layer_type,
                                                vllm_config, f'{prefix}.self_attn')

    def forward(self, x, positions, embeddings):
        normalized = self.operator_norm(x)
        if self.layer_type == 'linear_attention':
            mixed = self.conv(normalized, positions)
        else:
            mixed = self.self_attn(normalized, embeddings[self.layer_type])
        x = x + mixed
        return x + self.feed_forward(self.ffn_norm(x))


@support_torch_compile
class GretaBackbone(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ''):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim, config.pad_token_id)
        self.layers = nn.ModuleList([
            GretaLayer(config, i, vllm_config, f'{prefix}.layers.{i}')
            for i in range(config.num_hidden_layers)
        ])
        self.norm = GretaRMSNorm(config.hidden_dim, config.norm_eps)
        self.rotary_emb = GretaRotaryEmbedding(config, config.rope_theta)
        self.short_rotary_emb = GretaRotaryEmbedding(config, config.short_rope_theta)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        embeddings = {
            'full_attention': self.rotary_emb(x, positions.unsqueeze(0)),
            'sliding_attention': self.short_rotary_emb(x, positions.unsqueeze(0)),
        }
        for layer in self.layers:
            x = layer(x, positions, embeddings)
        return self.norm(x)


class GretaForCausalLM(nn.Module, HasInnerState, IsHybrid):
    """vLLM model; HF checkpoint names and tensor shapes are kept unchanged."""
    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config):
        return MambaStateDtypeCalculator.short_conv_state_dtype(
            vllm_config.model_config.dtype, vllm_config.cache_config.mamba_cache_dtype)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config):
        c = vllm_config.model_config.hf_config
        return MambaStateShapeCalculator.short_conv_state_shape(
            tp_world_size=1, intermediate_size=c.hidden_dim, conv_kernel=c.kernel_size)

    @classmethod
    def get_mamba_state_copy_func(cls):
        return MambaStateCopyFuncCalculator.short_conv_state_copy_func()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ''):
        super().__init__()
        validate_config(vllm_config)
        self.config = vllm_config.model_config.hf_config
        model_prefix = f'{prefix}.model' if prefix else 'model'
        self.model = GretaBackbone(vllm_config=vllm_config, prefix=model_prefix)
        self.lm_head = nn.Linear(self.config.hidden_dim, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def embed_input_ids(self, input_ids):
        return self.model.embed_tokens(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors=None,
                inputs_embeds=None, **kwargs):
        if intermediate_tensors is not None:
            raise ValueError('Pipeline parallelism is not supported')
        return self.model(input_ids, positions, None, inputs_embeds)

    def compute_logits(self, hidden_states):
        logits = self.lm_head(hidden_states).float()
        cap = self.config.logit_soft_cap
        return logits if cap is None else float(cap) * torch.tanh(logits / float(cap))

    @torch.no_grad()
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        tensors = dict(self.named_parameters(remove_duplicate=False))
        tensors.update(dict(self.named_buffers()))
        loaded = set()
        tied_source = None
        for name, weight in weights:
            if name not in tensors:
                raise ValueError(f'Unexpected Greta checkpoint tensor: {name}')
            if tensors[name].shape != weight.shape:
                raise ValueError(f'Shape mismatch for {name}: {weight.shape} != {tensors[name].shape}')
            target = tensors[name]
            if self.config.tie_word_embeddings and name in ('lm_head.weight', 'model.embed_tokens.weight'):
                if tied_source is not None and not torch.equal(target, weight.to(target)):
                    raise ValueError('Checkpoint has different tied embedding/head weights')
                tied_source = name
            target.copy_(weight)
            loaded.add(name)
        if tied_source is not None:
            loaded.update(('lm_head.weight', 'model.embed_tokens.weight'))
        # Legacy raw checkpoints may omit RoPE buffers; constructor computes them.
        # Learned tensors must never be silently left uninitialized.
        missing = set(dict(self.named_parameters())) - loaded
        if missing:
            raise ValueError(f'Missing Greta checkpoint parameters: {sorted(missing)}')
        return loaded
