"""Greta model configuration for Hugging Face Transformers."""

from __future__ import annotations

from typing import Iterable

from transformers.configuration_utils import PreTrainedConfig


# Transformers 5.17 normalizes recurrent/convolution layers to this name.
# The operator remains Greta dynamic convolution; legacy names are accepted below.
_DEFAULT_LAYER_TYPES = (
    "linear_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
) * 6

_LAYER_TYPE_ALIASES = {
    "conv": "linear_attention",
    "dynamic_convolution": "linear_attention",
    "dynamic_conv": "linear_attention",
    "attention": "full_attention",
}
_VALID_LAYER_TYPES = {"linear_attention", "sliding_attention", "full_attention"}
_VALID_ATTENTION_DTYPES = {"model", "float32", "bfloat16", "float16"}


class GretaConfig(PreTrainedConfig):
    """Configuration for the Greta hybrid causal language model."""

    model_type = "Greta"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 65_536,
        embedding_dim: int = 2_048,
        hidden_dim: int = 2_048,
        qk_dim: int = 128,
        v_dim: int = 128,
        num_q_heads: int = 16,
        num_kv_heads: int = 4,
        ffn_dim: int = 5_632,
        norm_eps: float = 1e-6,
        kernel_size: int = 7,
        num_k_heads: int = 128,
        max_position_embeddings: int = 32_768,
        rope_theta: float = 500_000.0,
        short_rope_theta: float = 10_000.0,
        rotary_factor: float = 0.5,
        sliding_window: int = 513,
        layer_types: Iterable[str] | None = None,
        attention_dropout: float = 0.0,
        initializer_range: float = 0.02,
        logit_soft_cap: float | None = 30.0,
        dynconv_norm: bool = True,
        dynconv_norm_eps: float = 1e-6,
        use_triton_kernels: bool = True,
        attention_compute_dtype: str = "bfloat16",
        use_cache: bool = True,
        tie_embeddings: bool | None = None,
        bos_token_id: int | None = 1,
        eos_token_id: int | list[int] | None = 65000,
        pad_token_id: int | None = 3,
        **kwargs,
    ):
        kwargs.pop("debug_check_finite", None)
        kwargs.pop("implementation_version", None)

        legacy_attention_backend = kwargs.pop("attention_backend", None)
        if kwargs.get("attn_implementation") is None and legacy_attention_backend is not None:
            if legacy_attention_backend not in {"eager", "sdpa"}:
                raise ValueError(
                    "Legacy attention_backend must be 'eager' or 'sdpa'; "
                    f"got {legacy_attention_backend!r}"
                )
            kwargs["attn_implementation"] = legacy_attention_backend

        # Keep the legacy name while allowing the standard HF option on its own.
        standard_tie_embeddings = kwargs.get("tie_word_embeddings")
        if tie_embeddings is None:
            tie_embeddings = True if standard_tie_embeddings is None else bool(standard_tie_embeddings)
        elif standard_tie_embeddings is not None and bool(standard_tie_embeddings) != bool(tie_embeddings):
            raise ValueError("tie_embeddings and tie_word_embeddings must agree")
        tie_embeddings = bool(tie_embeddings)

        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if embedding_dim <= 0 or hidden_dim <= 0:
            raise ValueError("embedding_dim and hidden_dim must be positive")
        if embedding_dim != hidden_dim:
            raise ValueError(
                "Greta requires embedding_dim == hidden_dim because there is no "
                "embedding-to-hidden projection."
            )
        if qk_dim <= 0 or v_dim <= 0:
            raise ValueError("qk_dim and v_dim must be positive")
        if num_q_heads <= 0 or num_kv_heads <= 0:
            raise ValueError("num_q_heads and num_kv_heads must be positive")
        if hidden_dim % num_q_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_q_heads")
        if num_q_heads % num_kv_heads != 0:
            raise ValueError("num_q_heads must be divisible by num_kv_heads")
        if ffn_dim <= 0:
            raise ValueError("ffn_dim must be positive")
        if norm_eps <= 0 or dynconv_norm_eps <= 0:
            raise ValueError("normalization eps values must be positive")
        if kernel_size < 1:
            raise ValueError("kernel_size must be >= 1")
        if num_k_heads <= 0:
            raise ValueError("num_k_heads must be positive")
        if hidden_dim % num_k_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_k_heads")
        if max_position_embeddings <= 0:
            raise ValueError("max_position_embeddings must be positive")
        if rope_theta <= 0 or short_rope_theta <= 0:
            raise ValueError("RoPE theta values must be positive")
        if not 0.0 < rotary_factor <= 1.0:
            raise ValueError("rotary_factor must be in (0, 1]")

        rotary_dim = int(qk_dim * rotary_factor)
        if rotary_dim <= 0 or rotary_dim % 2 != 0:
            raise ValueError("int(qk_dim * rotary_factor) must be a positive even number")
        if sliding_window < 1:
            raise ValueError("sliding_window must include at least the current token")
        if not 0.0 <= attention_dropout <= 1.0:
            raise ValueError("attention_dropout must be in [0, 1]")
        if initializer_range < 0:
            raise ValueError("initializer_range must be non-negative")
        if logit_soft_cap is not None and logit_soft_cap <= 0:
            raise ValueError("logit_soft_cap must be positive when set")
        if attention_compute_dtype not in _VALID_ATTENTION_DTYPES:
            raise ValueError(
                f"attention_compute_dtype must be one of {sorted(_VALID_ATTENTION_DTYPES)}, "
                f"got {attention_compute_dtype!r}"
            )
        layer_types = list(_DEFAULT_LAYER_TYPES if layer_types is None else layer_types)
        layer_types = [_LAYER_TYPE_ALIASES.get(layer_type, layer_type) for layer_type in layer_types]
        invalid_layer_types = sorted(set(layer_types) - _VALID_LAYER_TYPES)
        if invalid_layer_types:
            raise ValueError(
                f"Unsupported Greta layer types: {invalid_layer_types}. "
                f"Expected values from {sorted(_VALID_LAYER_TYPES)}."
            )
        if not layer_types:
            raise ValueError("layer_types must contain at least one layer")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.hidden_size = hidden_dim
        self.head_dim = qk_dim  # vLLM must not infer this from hidden_size / heads.
        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.num_attention_heads = num_q_heads
        self.num_key_value_heads = num_kv_heads
        self.ffn_dim = ffn_dim
        self.intermediate_size = ffn_dim
        self.norm_eps = norm_eps
        self.rms_norm_eps = norm_eps

        self.kernel_size = kernel_size
        self.num_k_heads = num_k_heads
        self.conv_group_size = num_k_heads
        self.num_conv_filters = hidden_dim // num_k_heads
        self.number_of_conv_states = 1

        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.short_rope_theta = short_rope_theta
        self.rotary_factor = rotary_factor
        self.rotary_dim = rotary_dim
        self.sliding_window = sliding_window
        self.sliding_window_left = sliding_window - 1

        self.layer_types = layer_types
        self.num_hidden_layers = len(layer_types)
        # vLLM's hybrid layer accounting uses this explicit map. Both local
        # and global attention count as attention; conv uses its Mamba cache API.
        self.layers_block_type = [
            "mamba" if kind == "linear_attention" else "attention"
            for kind in layer_types
        ]
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.logit_soft_cap = logit_soft_cap
        self.dynconv_norm = dynconv_norm
        self.dynconv_norm_eps = dynconv_norm_eps
        self.use_triton_kernels = use_triton_kernels
        self.attention_compute_dtype = attention_compute_dtype
        self.use_cache = use_cache
        self.tie_embeddings = tie_embeddings

        kwargs.setdefault("is_decoder", True)
        kwargs.setdefault("is_encoder_decoder", False)
        kwargs.setdefault("tie_word_embeddings", tie_embeddings)

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )


GretaConfig.register_for_auto_class()
