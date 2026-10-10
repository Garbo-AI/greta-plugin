"""Fused dynamic causal depthwise convolution kernels.

The training path uses a physical batch size of one with ``cu_seqlens`` to
represent packed variable-length sequences. The inference path accepts regular
batched context tensors and is forward-only. Both paths implement compact
channel-group filter sharing, optional per-group RMS normalization, and an
optional SiLU output gate.

For a token ``t`` and channel ``d``:

    y[t, d] = sum_k w[t, k, d // G] * x[t + k - W + 1, d]
    out[t,d] = RMSNorm_group(y[t,d]) * SiLU(g[t,d])

when normalization and gating are enabled.

With group RMS normalization enabled, a per-token/per-filter scalar rescaling
of all taps is divided out by the normalization, so the kernel consumes raw
dynamic weights without an additional normalization step.
"""

import contextlib
import functools
import inspect
from typing import Optional

import torch
import torch.nn.functional as F

import triton
import triton.language as tl

from einops import rearrange

device_torch_lib = getattr(torch, "cuda")

NUM_WARPS_AUTOTUNE = [4, 8, 16, 32]
BD_AUTOTUNE = [16, 32, 64, 128]


def custom_device_ctx(index: int):
    return device_torch_lib.device(index)


def input_guard(fn=None, *, no_guard_contiguous=False):
    def decorator(fn):
        sig = inspect.signature(fn)
        param_names = list(sig.parameters.keys())

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            skip_params = set()
            if isinstance(no_guard_contiguous, list):
                skip_params = set(no_guard_contiguous)

            processed_args = []
            for i, arg in enumerate(args):
                param_name = param_names[i] if i < len(param_names) else f"__arg_{i}"
                if isinstance(arg, torch.Tensor):
                    if no_guard_contiguous is True or param_name in skip_params:
                        processed_args.append(arg)
                    else:
                        processed_args.append(arg.contiguous())
                else:
                    processed_args.append(arg)

            processed_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    if no_guard_contiguous is True or k in skip_params:
                        processed_kwargs[k] = v
                    else:
                        processed_kwargs[k] = v.contiguous()
                else:
                    processed_kwargs[k] = v

            tensor = None
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    tensor = arg
                    break
            if tensor is None:
                for value in kwargs.values():
                    if isinstance(value, torch.Tensor):
                        tensor = value
                        break

            ctx = custom_device_ctx(tensor.device.index) if tensor is not None and tensor.is_cuda \
                else contextlib.nullcontext()
            with ctx:
                return fn(*processed_args, **processed_kwargs)

        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator


def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return torch.diff(cu_seqlens)


def prepare_chunk_indices(
    cu_seqlens: torch.Tensor,
    chunk_size: int,
    cu_seqlens_cpu: torch.Tensor | None = None,
) -> torch.Tensor:
    # Pass cu_seqlens_cpu to avoid a GPU->CPU sync here.
    src = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens
    indices = torch.cat(
        [torch.arange(n) for n in triton.cdiv(prepare_lens(src), chunk_size).tolist()]
    )
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


@functools.cache
def _get_multiprocessor_count_impl(tensor_idx: int = 0) -> int:
    try:
        props = triton.runtime.driver.active.utils.get_device_properties(tensor_idx)
        return props["multiprocessor_count"]
    except BaseException:
        try:
            if triton.runtime.driver.active.get_current_target().backend == "npu":
                props = triton.runtime.driver.active.utils.get_device_properties(tensor_idx)
                return props["num_vectorcore"]
        except BaseException:
            pass
        return 1


@torch.compiler.assume_constant_result
def get_multiprocessor_count(tensor_idx: int = 0) -> int:
    return _get_multiprocessor_count_impl(int(tensor_idx or 0))


def _prune_invalid_bd(configs, named_args, **kwargs):
    """Keep only configs whose channel block BD is a multiple of the group size G,
    so the in-register broadcast/reduce reshape is exact.

    Depending on the Triton version, launch arguments arrive in `named_args`
    (positional) or in `**kwargs` (keyword launches), so check both."""
    args = {**(named_args or {}), **kwargs}
    G = args.get("G")
    if G is None:
        raise RuntimeError(
            "could not find 'G' in autotune prune arguments; "
            f"named_args={list((named_args or {}).keys())}, kwargs={list(kwargs.keys())}"
        )
    pruned = [cfg for cfg in configs if cfg.kwargs["BD"] % G == 0]
    if not pruned:
        raise ValueError(
            f"group_size={G} must divide one of BD={BD_AUTOTUNE} "
            f"(i.e. be a power of two <= {max(BD_AUTOTUNE)})."
        )
    return pruned

@triton.jit
def _load_2d(base, row_offsets, col_offsets, row_stride, row_limit, col_limit):
    """Load a 2D tile with zero padding using ordinary pointer tensors."""
    rows = row_offsets[:, None]
    cols = col_offsets[None, :]
    mask = (rows >= 0) & (rows < row_limit) & (cols >= 0) & (cols < col_limit)
    return tl.load(base + rows * row_stride + cols, mask=mask, other=0.0)


@triton.jit
def _store_2d(base, row_offsets, col_offsets, row_stride, row_limit, col_limit, values):
    """Store a 2D tile with the same bounds semantics as the old block pointers."""
    rows = row_offsets[:, None]
    cols = col_offsets[None, :]
    mask = (rows >= 0) & (rows < row_limit) & (cols >= 0) & (cols < col_limit)
    values = tl.cast(values, dtype=base.dtype.element_ty, fp_downcast_rounding="rtne")
    tl.store(base + rows * row_stride + cols, values, mask=mask)


# ---------------------------------------------------------------------------
# Forward kernel
# ---------------------------------------------------------------------------
@triton.heuristics({
    "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    "HAS_GATE": lambda args: args["g"] is not None,
    "SAVE_PRE": lambda args: args["yr"] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({"BD": BD}, num_warps=num_warps)
        for BD in BD_AUTOTUNE
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=["D", "W", "G", "BT", "HAS_NORM"],
    prune_configs_by={"early_config_prune": _prune_invalid_bd},
)
@triton.jit
def dynconv_fwd_kernel(
    x,                # [1, T, D]
    w,                # [T, W, D // G]   compact, tap-major
    g,                # [1, T, D] or None
    y,                # [1, T, D]        final output (normed/gated)
    yr,               # [1, T, D] or None: pre-norm conv output, saved for bwd
    cu_seqlens, chunk_indices,
    T,
    eps,
    D: tl.constexpr,
    G: tl.constexpr,  # channels per shared filter == channels per norm group
    W: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    HAS_NORM: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HAS_GATE: tl.constexpr,
    SAVE_PRE: tl.constexpr,
):
    tl.static_assert(BD % G == 0)
    BDC: tl.constexpr = BD // G
    DC: tl.constexpr = D // G

    i_t, i_d, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = i_b.to(tl.int64) * T

    offs_t = i_t * BT + tl.arange(0, BT)
    offs_d = i_d * BD + tl.arange(0, BD)
    offs_c = i_d * BDC + tl.arange(0, BDC)

    b_y = tl.zeros((BT, BD), dtype=tl.float32)
    for i_w in tl.static_range(-W + 1, 1):
        k = i_w + W - 1
        b_x = _load_2d(x + bos * D, offs_t + i_w, offs_d, D, T, D).to(tl.float32)
        b_w = _load_2d(
            w + bos * W * DC + k * DC,
            offs_t,
            offs_c,
            W * DC,
            T,
            DC,
        ).to(tl.float32)
        b_wf = tl.reshape(tl.broadcast_to(b_w[:, :, None], (BT, BDC, G)), (BT, BD))
        b_y += b_x * b_wf

    if HAS_NORM:
        if SAVE_PRE:
            _store_2d(yr + bos * D, offs_t, offs_d, D, T, D, b_y)

        b_ms = tl.sum(tl.reshape(b_y * b_y, (BT, BDC, G)), axis=2) / G
        b_rinv = tl.reshape(
            tl.broadcast_to((1.0 / tl.sqrt(b_ms + eps))[:, :, None], (BT, BDC, G)), (BT, BD)
        )
        b_y = b_y * b_rinv

    if HAS_GATE:
        b_g = _load_2d(g + bos * D, offs_t, offs_d, D, T, D).to(tl.float32)
        b_y = b_y * b_g * tl.sigmoid(b_g)

    _store_2d(y + bos * D, offs_t, offs_d, D, T, D, b_y)



# ---------------------------------------------------------------------------
# Backward kernel: dx, compact dw, dg in one pass.
#   norm off: pre-gate y recomputed on the fly (nothing saved)
#   norm on : pre-norm y loaded from `yr` (saved by forward)
# ---------------------------------------------------------------------------

@triton.heuristics({
    "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    "HAS_GATE": lambda args: args["g"] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({"BD": BD}, num_warps=num_warps)
        for BD in BD_AUTOTUNE
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=["D", "W", "G", "BT", "HAS_NORM"],
    prune_configs_by={"early_config_prune": _prune_invalid_bd},
)
@triton.jit
def dynconv_bwd_kernel(
    x, w, g, yr, dout,
    dx, dw, dg,
    cu_seqlens, chunk_indices,
    T,
    eps,
    D: tl.constexpr,
    G: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    HAS_NORM: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HAS_GATE: tl.constexpr,
):
    tl.static_assert(BD % G == 0)
    BDC: tl.constexpr = BD // G
    DC: tl.constexpr = D // G

    i_t, i_d, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = i_b.to(tl.int64) * T

    offs_t = i_t * BT + tl.arange(0, BT)
    offs_d = i_d * BD + tl.arange(0, BD)
    offs_c = i_d * BDC + tl.arange(0, BDC)

    b_do = _load_2d(dout + bos * D, offs_t, offs_d, D, T, D).to(tl.float32)

    if HAS_GATE:
        b_g = _load_2d(g + bos * D, offs_t, offs_d, D, T, D).to(tl.float32)
        b_sig = tl.sigmoid(b_g)
        b_dn = b_do * b_g * b_sig
    else:
        b_dn = b_do

    if HAS_NORM:
        b_yp = _load_2d(yr + bos * D, offs_t, offs_d, D, T, D).to(tl.float32)
        b_ms = tl.sum(tl.reshape(b_yp * b_yp, (BT, BDC, G)), axis=2) / G
        b_rinv = tl.reshape(
            tl.broadcast_to((1.0 / tl.sqrt(b_ms + eps))[:, :, None], (BT, BDC, G)), (BT, BD)
        )
        b_n = b_yp * b_rinv
        b_c = tl.reshape(
            tl.broadcast_to(
                (tl.sum(tl.reshape(b_n * b_dn, (BT, BDC, G)), axis=2) / G)[:, :, None],
                (BT, BDC, G),
            ),
            (BT, BD),
        )
        b_dy = b_rinv * (b_dn - b_n * b_c)
    else:
        b_dy = b_dn

    b_dx = tl.zeros((BT, BD), dtype=tl.float32)
    b_yc = tl.zeros((BT, BD), dtype=tl.float32)

    for i_w in tl.static_range(0, W):
        k_dx = W - 1 - i_w

        b_do_s = _load_2d(dout + bos * D, offs_t + i_w, offs_d, D, T, D).to(tl.float32)
        if HAS_GATE:
            b_g_s = _load_2d(g + bos * D, offs_t + i_w, offs_d, D, T, D).to(tl.float32)
            b_dy_s = b_do_s * b_g_s * tl.sigmoid(b_g_s)
        else:
            b_dy_s = b_do_s

        if HAS_NORM:
            b_yp_s = _load_2d(yr + bos * D, offs_t + i_w, offs_d, D, T, D).to(tl.float32)
            b_ms_s = tl.sum(tl.reshape(b_yp_s * b_yp_s, (BT, BDC, G)), axis=2) / G
            b_rv_s = tl.reshape(
                tl.broadcast_to((1.0 / tl.sqrt(b_ms_s + eps))[:, :, None], (BT, BDC, G)),
                (BT, BD),
            )
            b_n_s = b_yp_s * b_rv_s
            b_c_s = tl.reshape(
                tl.broadcast_to(
                    (tl.sum(tl.reshape(b_n_s * b_dy_s, (BT, BDC, G)), axis=2) / G)[:, :, None],
                    (BT, BDC, G),
                ),
                (BT, BD),
            )
            b_dy_s = b_rv_s * (b_dy_s - b_n_s * b_c_s)

        b_w_s = _load_2d(
            w + bos * W * DC + k_dx * DC,
            offs_t + i_w,
            offs_c,
            W * DC,
            T,
            DC,
        ).to(tl.float32)
        b_wf_s = tl.reshape(tl.broadcast_to(b_w_s[:, :, None], (BT, BDC, G)), (BT, BD))
        b_dx += b_dy_s * b_wf_s

        b_x_s = _load_2d(
            x + bos * D,
            offs_t + i_w - W + 1,
            offs_d,
            D,
            T,
            D,
        ).to(tl.float32)
        b_dw_full = b_dy * b_x_s
        b_dw = tl.sum(tl.reshape(b_dw_full, (BT, BDC, G)), axis=2)
        _store_2d(
            dw + bos * W * DC + i_w * DC,
            offs_t,
            offs_c,
            W * DC,
            T,
            DC,
            b_dw,
        )

        if HAS_GATE:
            if not HAS_NORM:
                b_w_t = _load_2d(
                    w + bos * W * DC + i_w * DC,
                    offs_t,
                    offs_c,
                    W * DC,
                    T,
                    DC,
                ).to(tl.float32)
                b_wf_t = tl.reshape(tl.broadcast_to(b_w_t[:, :, None], (BT, BDC, G)), (BT, BD))
                b_yc += b_x_s * b_wf_t

    _store_2d(dx + bos * D, offs_t, offs_d, D, T, D, b_dx)

    if HAS_GATE:
        if HAS_NORM:
            b_hat = b_n
        else:
            b_hat = b_yc
        b_dg = b_do * b_hat * (b_sig * (1.0 + b_g * (1.0 - b_sig)))
        _store_2d(dg + bos * D, offs_t, offs_d, D, T, D, b_dg)



# ---------------------------------------------------------------------------
# Batched inference kernel
# ---------------------------------------------------------------------------
@triton.heuristics({
    "HAS_GATE": lambda args: args["g"] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({"BD": BD}, num_warps=num_warps)
        for BD in BD_AUTOTUNE
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=["D", "W", "G", "BT", "HAS_NORM"],
    prune_configs_by={"early_config_prune": _prune_invalid_bd},
)
@triton.jit
def dynconv_inference_kernel(
    context,           # [B, Q + W - 1, D]
    w,                 # [B, Q, W, D // G]
    g,                 # [B, Q, D] or None
    y,                 # [B, Q, D]
    Q,
    eps,
    D: tl.constexpr,
    G: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    HAS_NORM: tl.constexpr,
    HAS_GATE: tl.constexpr,
):
    tl.static_assert(BD % G == 0)
    BDC: tl.constexpr = BD // G
    DC: tl.constexpr = D // G

    i_t, i_d, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    context_length = Q + W - 1
    offs_t = i_t * BT + tl.arange(0, BT)
    offs_d = i_d * BD + tl.arange(0, BD)
    offs_c = i_d * BDC + tl.arange(0, BDC)

    b_y = tl.zeros((BT, BD), dtype=tl.float32)
    for k in tl.static_range(0, W):
        b_x = _load_2d(
            context + i_b * context_length * D + k * D,
            offs_t,
            offs_d,
            D,
            Q,
            D,
        ).to(tl.float32)
        b_w = _load_2d(
            w + i_b * Q * W * DC + k * DC,
            offs_t,
            offs_c,
            W * DC,
            Q,
            DC,
        ).to(tl.float32)
        b_wf = tl.reshape(tl.broadcast_to(b_w[:, :, None], (BT, BDC, G)), (BT, BD))
        b_y += b_x * b_wf

    if HAS_NORM:
        b_ms = tl.sum(tl.reshape(b_y * b_y, (BT, BDC, G)), axis=2) / G
        b_rinv = tl.reshape(
            tl.broadcast_to((1.0 / tl.sqrt(b_ms + eps))[:, :, None], (BT, BDC, G)),
            (BT, BD),
        )
        b_y = b_y * b_rinv

    if HAS_GATE:
        b_g = _load_2d(g + i_b * Q * D, offs_t, offs_d, D, Q, D).to(tl.float32)
        b_y = b_y * b_g * tl.sigmoid(b_g)

    _store_2d(y + i_b * Q * D, offs_t, offs_d, D, Q, D, b_y)



# ---------------------------------------------------------------------------
# Single-token cached decode kernel
# ---------------------------------------------------------------------------
@triton.heuristics({
    "HAS_GATE": lambda args: args["g"] is not None,
})
@triton.jit
def dynconv_decode_kernel(
    x,                  # [B, D] current projected input
    w,                  # [B, W, D // G]
    g,                  # [B, D] or None
    conv_state,         # [B, D, W - 1], updated in-place
    y,                  # [B, D]
    eps,
    D: tl.constexpr,
    G: tl.constexpr,
    W: tl.constexpr,
    BD: tl.constexpr,
    HAS_NORM: tl.constexpr,
    HAS_GATE: tl.constexpr,
):
    tl.static_assert(BD % G == 0)
    H: tl.constexpr = W - 1
    BDC: tl.constexpr = BD // G
    DC: tl.constexpr = D // G

    i_d, i_b = tl.program_id(0), tl.program_id(1)
    offs_d = i_d * BD + tl.arange(0, BD)
    offs_c = i_d * BDC + tl.arange(0, BDC)
    mask_d = offs_d < D
    mask_c = offs_c < DC

    b_y = tl.zeros((BD,), dtype=tl.float32)
    for k in tl.static_range(0, H):
        b_x = tl.load(
            conv_state + i_b * D * H + offs_d * H + k,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        b_w = tl.load(
            w + i_b * W * DC + k * DC + offs_c,
            mask=mask_c,
            other=0.0,
        ).to(tl.float32)
        b_wf = tl.reshape(tl.broadcast_to(b_w[:, None], (BDC, G)), (BD,))
        b_y += b_x * b_wf

    b_x = tl.load(x + i_b * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    b_w = tl.load(
        w + i_b * W * DC + H * DC + offs_c,
        mask=mask_c,
        other=0.0,
    ).to(tl.float32)
    b_wf = tl.reshape(tl.broadcast_to(b_w[:, None], (BDC, G)), (BD,))
    b_y += b_x * b_wf

    if HAS_NORM:
        b_ms = tl.sum(tl.reshape(b_y * b_y, (BDC, G)), axis=1) / G
        b_rinv = tl.reshape(
            tl.broadcast_to((1.0 / tl.sqrt(b_ms + eps))[:, None], (BDC, G)),
            (BD,),
        )
        b_y *= b_rinv

    if HAS_GATE:
        b_g = tl.load(g + i_b * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
        b_y *= b_g * tl.sigmoid(b_g)

    tl.store(
        y + i_b * D + offs_d,
        b_y,
        mask=mask_d,
    )

    # Shift the fixed-size state left and append the current projected input.
    # Sources are always one position to the right of destinations, so no source
    # is overwritten before it is read.
    for k in tl.static_range(0, H - 1):
        b_next = tl.load(
            conv_state + i_b * D * H + offs_d * H + k + 1,
            mask=mask_d,
            other=0.0,
        )
        tl.store(
            conv_state + i_b * D * H + offs_d * H + k,
            b_next,
            mask=mask_d,
        )
    tl.store(
        conv_state + i_b * D * H + offs_d * H + H - 1,
        b_x,
        mask=mask_d,
    )


# ---------------------------------------------------------------------------
# Python launchers
# ---------------------------------------------------------------------------
def _chunk_bt(T: int, B: int, device_index: int) -> int:
    return min(64, triton.next_power_of_2(
        triton.cdiv(max(16, B * T), get_multiprocessor_count(device_index))
    ))


def _check_inference_shapes(context, weight, gate, group_size):
    if context.dim() != 3:
        raise ValueError(f"context must be [B, Q + W - 1, D], got {tuple(context.shape)}")
    if weight.dim() != 4:
        raise ValueError(f"weight must be [B, Q, W, D // group_size], got {tuple(weight.shape)}")

    batch_size, context_length, hidden_size = context.shape
    weight_batch, query_length, kernel_size, num_filters = weight.shape
    if batch_size != weight_batch:
        raise ValueError(f"context batch={batch_size} does not match weight batch={weight_batch}")
    if hidden_size % group_size != 0:
        raise ValueError(f"group_size={group_size} must divide hidden_size={hidden_size}")
    if num_filters != hidden_size // group_size:
        raise ValueError(
            f"weight has {num_filters} filters, expected hidden_size // group_size="
            f"{hidden_size // group_size}"
        )
    if context_length != query_length + kernel_size - 1:
        raise ValueError(
            "context must contain exactly W-1 history tokens followed by the query chunk; "
            f"got context_length={context_length}, query_length={query_length}, kernel_size={kernel_size}"
        )
    if gate is not None and gate.shape != (batch_size, query_length, hidden_size):
        raise ValueError(
            f"gate must be [{batch_size}, {query_length}, {hidden_size}], got {tuple(gate.shape)}"
        )
    return batch_size, query_length, hidden_size, kernel_size


@torch.compiler.disable(recursive=True)
def dynamic_causal_conv1d_inference(
    context: torch.Tensor,
    weight: torch.Tensor,
    gate: Optional[torch.Tensor] = None,
    group_size: int = 1,
    norm: bool = False,
    norm_eps: float = 1e-6,
) -> torch.Tensor:
    """Batched forward-only dynamic convolution for Hugging Face inference.

    `context` contains `W - 1` left-context tokens followed by the `Q` query tokens.
    Unlike `dynamic_causal_conv1d`, this path is not packed and supports `B > 1`.
    """
    if not context.is_cuda or not weight.is_cuda or (gate is not None and not gate.is_cuda):
        raise ValueError("dynamic_causal_conv1d_inference requires CUDA tensors")

    context = context.contiguous()
    weight = weight.contiguous()
    gate = gate.contiguous() if gate is not None else None
    B, Q, D, W = _check_inference_shapes(context, weight, gate, group_size)
    if Q == 0:
        return context.new_empty((B, 0, D))

    BT = _chunk_bt(Q, B, context.device.index)
    y = context.new_empty((B, Q, D))

    def grid(meta):
        return (triton.cdiv(Q, BT), triton.cdiv(D, meta["BD"]), B)

    with custom_device_ctx(context.device.index):
        dynconv_inference_kernel[grid](
            context=context,
            w=weight,
            g=gate,
            y=y,
            Q=Q,
            eps=norm_eps,
            D=D,
            G=group_size,
            W=W,
            BT=BT,
            HAS_NORM=norm,
        )
    return y


def _check_decode_shapes(x, weight, gate, conv_state, group_size):
    if x.dim() != 2:
        raise ValueError(f"x must be [B, D], got {tuple(x.shape)}")
    if weight.dim() != 3:
        raise ValueError(f"weight must be [B, W, D // group_size], got {tuple(weight.shape)}")
    if conv_state.dim() != 3:
        raise ValueError(f"conv_state must be [B, D, W - 1], got {tuple(conv_state.shape)}")

    batch_size, hidden_size = x.shape
    weight_batch, kernel_size, num_filters = weight.shape
    if kernel_size < 2:
        raise ValueError("cached decode requires kernel_size >= 2")
    if weight_batch != batch_size:
        raise ValueError(f"x batch={batch_size} does not match weight batch={weight_batch}")
    if hidden_size % group_size != 0:
        raise ValueError(f"group_size={group_size} must divide hidden_size={hidden_size}")
    if num_filters != hidden_size // group_size:
        raise ValueError(
            f"weight has {num_filters} filters, expected {hidden_size // group_size}"
        )
    expected_state = (batch_size, hidden_size, kernel_size - 1)
    if tuple(conv_state.shape) != expected_state:
        raise ValueError(f"conv_state must be {expected_state}, got {tuple(conv_state.shape)}")
    if gate is not None and gate.shape != x.shape:
        raise ValueError(f"gate shape {tuple(gate.shape)} must match x {tuple(x.shape)}")
    if 128 % group_size != 0:
        raise ValueError(
            f"cached decode currently requires group_size to divide 128, got {group_size}"
        )
    return batch_size, hidden_size, kernel_size


@torch.compiler.disable(recursive=True)
def dynamic_causal_conv1d_decode(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    gate: Optional[torch.Tensor] = None,
    group_size: int = 1,
    norm: bool = False,
    norm_eps: float = 1e-6,
) -> torch.Tensor:
    """Single-token cached decode with an in-place fixed-size convolution state update.

    `x` is the current projected token `[B, D]`; `conv_state` contains the previous
    `W - 1` projected inputs in chronological order and is updated in-place.
    """
    if not x.is_cuda or not weight.is_cuda or not conv_state.is_cuda or (gate is not None and not gate.is_cuda):
        raise ValueError("dynamic_causal_conv1d_decode requires CUDA tensors")

    x = x.contiguous()
    weight = weight.contiguous()
    gate = gate.contiguous() if gate is not None else None
    if not conv_state.is_contiguous():
        raise ValueError("conv_state must be contiguous because it is updated in-place")

    B, D, W = _check_decode_shapes(x, weight, gate, conv_state, group_size)
    y = torch.empty_like(x)
    BD = 128

    with custom_device_ctx(x.device.index):
        dynconv_decode_kernel[(triton.cdiv(D, BD), B)](
            x=x,
            w=weight,
            g=gate,
            conv_state=conv_state,
            y=y,
            eps=norm_eps,
            D=D,
            G=group_size,
            W=W,
            BD=BD,
            HAS_NORM=norm,
            num_warps=4,
        )
    return y


def _check_shapes(x, weight, gate, group_size):
    if x.dim() != 3:
        raise ValueError(f"x must be [1, T, D], got {tuple(x.shape)}")
    B, T, D = x.shape
    if B != 1:
        raise ValueError("the packed dynamic-convolution training kernel requires physical batch size B=1")
    if D % group_size != 0:
        raise ValueError(f"group_size={group_size} must divide D={D}")
    DC = D // group_size
    if weight.dim() != 3 or weight.shape[0] != T or weight.shape[2] != DC:
        raise ValueError(
            f"weight must be [T, W, D // group_size] = [{T}, *, {DC}], got {tuple(weight.shape)}"
        )
    if gate is not None and gate.shape != x.shape:
        raise ValueError(f"gate shape {tuple(gate.shape)} must match x {tuple(x.shape)}")
    return B, T, D, weight.shape[1]


def dynconv_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    gate: Optional[torch.Tensor],
    cu_seqlens: Optional[torch.Tensor],
    group_size: int,
    norm: bool = False,
    eps: float = 1e-6,
    save_pre: bool = False,
    BT: Optional[int] = None,
    chunk_indices: Optional[torch.Tensor] = None,
    cu_seqlens_cpu: Optional[torch.Tensor] = None,
):
    shape_x = x.shape
    if x.dim() > 3:
        x = rearrange(x, "b t ... -> b t (...)")
        if gate is not None and gate.dim() > 3:
            gate = rearrange(gate, "b t ... -> b t (...)")
    B, T, D, W = _check_shapes(x, weight, gate, group_size)

    if BT is None:
        BT = _chunk_bt(T, B, x.device.index)
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT, cu_seqlens_cpu)
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)

    y = torch.empty_like(x)
    yr = torch.empty_like(x) if (norm and save_pre) else None

    def grid(meta):
        return (NT, triton.cdiv(D, meta["BD"]), B)

    dynconv_fwd_kernel[grid](
        x=x, w=weight, g=gate, y=y, yr=yr,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        T=T, eps=eps, D=D, G=group_size, W=W, BT=BT, HAS_NORM=norm,
    )
    return y.view(shape_x), yr, BT, chunk_indices


def dynconv_bwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    gate: Optional[torch.Tensor],
    yr: Optional[torch.Tensor],
    dout: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor],
    group_size: int,
    norm: bool,
    eps: float,
    BT: int,
    chunk_indices: Optional[torch.Tensor],
):
    shape_x = x.shape
    if x.dim() > 3:
        x = rearrange(x, "b t ... -> b t (...)")
        dout = rearrange(dout, "b t ... -> b t (...)")
        if gate is not None and gate.dim() > 3:
            gate = rearrange(gate, "b t ... -> b t (...)")
    B, T, D, W = _check_shapes(x, weight, gate, group_size)
    if norm and yr is None:
        raise RuntimeError("norm=True backward requires the saved pre-norm output")

    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)

    dx = torch.empty_like(x)
    dw = torch.empty_like(weight)
    dg = torch.empty_like(gate) if gate is not None else None

    def grid(meta):
        return (NT, triton.cdiv(D, meta["BD"]), B)

    dynconv_bwd_kernel[grid](
        x=x, w=weight, g=gate, yr=yr, dout=dout,
        dx=dx, dw=dw, dg=dg,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        T=T, eps=eps, D=D, G=group_size, W=W, BT=BT, HAS_NORM=norm,
    )
    return dx.view(shape_x), dw, dg


class DynamicCausalConv1dFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(ctx, x, weight, gate, cu_seqlens, group_size, norm, eps, cu_seqlens_cpu=None):
        y, yr, BT, chunk_indices = dynconv_fwd(
            x, weight, gate, cu_seqlens, group_size,
            norm=norm, eps=eps, save_pre=True, cu_seqlens_cpu=cu_seqlens_cpu,
        )
        ctx.save_for_backward(x, weight, gate, yr)
        ctx.cu_seqlens = cu_seqlens
        ctx.chunk_indices = chunk_indices   # reuse in bwd: no recompute, no extra sync
        ctx.BT = BT
        ctx.group_size = group_size
        ctx.norm = norm
        ctx.eps = eps
        return y

    @staticmethod
    @input_guard
    def backward(ctx, dout):
        x, weight, gate, yr = ctx.saved_tensors
        dx, dw, dg = dynconv_bwd(
            x, weight, gate, yr, dout,
            ctx.cu_seqlens, ctx.group_size, ctx.norm, ctx.eps,
            ctx.BT, ctx.chunk_indices,
        )
        return dx, dw, dg, None, None, None, None, None


@torch.compiler.disable(recursive=True)
def dynamic_causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    gate: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    group_size: int = 1,
    norm: bool = False,
    norm_eps: float = 1e-6,
    cu_seqlens_cpu: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    x:      [1, T, D]
    weight: [T, W, D // group_size]   (tap-major, compact; channel j uses filter j // group_size)
    gate:   [1, T, D] or None
    norm:   per-group (G channels) weightless RMSNorm on the conv output, before the gate

    out = RMSNorm_group(conv(x, w)) * silu(gate)
    """
    needs_grad = torch.is_grad_enabled() and (
        x.requires_grad or weight.requires_grad
        or (gate is not None and gate.requires_grad)
    )
    if needs_grad:
        return DynamicCausalConv1dFunction.apply(
            x, weight, gate, cu_seqlens, group_size, norm, norm_eps, cu_seqlens_cpu
        )
    # inference: same kernel, no autograd bookkeeping, nothing saved
    with custom_device_ctx(x.device.index):
        y, _, _, _ = dynconv_fwd(
            x.contiguous(), weight.contiguous(),
            gate.contiguous() if gate is not None else None,
            cu_seqlens, group_size,
            norm=norm, eps=norm_eps, save_pre=False, cu_seqlens_cpu=cu_seqlens_cpu,
        )
    return y


# ---------------------------------------------------------------------------
# Reference implementation (for testing / CPU)
# ---------------------------------------------------------------------------
def torch_dynamic_causal_conv1d(
    x: torch.Tensor,            # [1, T, D]
    weight: torch.Tensor,       # [T, W, D // G] compact
    gate: Optional[torch.Tensor],
    cu_seqlens: torch.Tensor,
    group_size: int,
    norm: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    B, T, D = x.shape
    W = weight.shape[1]
    w_full = weight.repeat_interleave(group_size, dim=2)  # [T, W, D]
    outs = []
    for s, e in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
        xs = x[0, s:e]                               # [L, D]
        ws = w_full[s:e]                             # [L, W, D]
        xp = F.pad(xs, (0, 0, W - 1, 0))             # [L + W - 1, D]
        xu = xp.unfold(0, W, 1)                      # [L, D, W]; xu[t,d,k] = x[t + k - W + 1, d]
        outs.append((xu * ws.permute(0, 2, 1)).sum(-1))
    y = torch.cat(outs, 0).unsqueeze(0)
    if norm:
        yg = y.float().view(1, T, D // group_size, group_size)
        y = (yg * torch.rsqrt(yg.pow(2).mean(-1, keepdim=True) + eps)).view(1, T, D).to(y.dtype)
    if gate is not None:
        y = y * F.silu(gate)
    return y


__all__ = [
    "dynamic_causal_conv1d",
    "dynamic_causal_conv1d_decode",
    "dynamic_causal_conv1d_inference",
]
