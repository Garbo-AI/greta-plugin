"""Triton kernels for Greta's dynamic convolution over vLLM's packed batches.

A batch is a flat run of T tokens from N requests (``starts`` holds the
cumulative offsets, decodes first). Each request owns a cache slot holding its
last ``W - 1`` projected inputs. Two launches per layer:

1. ``_conv_kernel``: every (token block, channel block) program computes the
   convolution for its tokens, taking history from earlier tokens of the same
   request in this batch or from the request's slot, then applies the group
   RMS norm and the SiLU gate. Tokens are independent, so there is no serial
   scan; the only state read is the slot, which is not written here.
2. ``_state_kernel``: per (request, channel block), writes the request's last
   ``W - 1`` inputs back to its slot. Running after (1) on the same stream keeps
   the slot reads of (1) ahead of these writes.

Math matches ``greta.modeling_greta.torch_dynamic_causal_conv1d_from_context``:
fp32 accumulation, the overflow-safe group RMS norm (``_stable_group_rms_norm``),
then ``silu(gate)``. The gate is applied in fp32 before the single rounding.

Launches use only device-side values and tensor shapes, so a decode step can be
captured in a CUDA graph.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _conv_kernel(
    x, w, g, y, state,
    starts, seq_idx, slots, initial,
    T, eps,
    s_slot, s_d, s_h,
    D: tl.constexpr, G: tl.constexpr, W: tl.constexpr,
    BT: tl.constexpr, BD: tl.constexpr,
    HAS_NORM: tl.constexpr, HAS_GATE: tl.constexpr,
):
    H: tl.constexpr = W - 1
    BDC: tl.constexpr = BD // G
    DC: tl.constexpr = D // G
    pid_t, pid_d = tl.program_id(0), tl.program_id(1)

    offs_t = pid_t * BT + tl.arange(0, BT)
    offs_d = pid_d * BD + tl.arange(0, BD)
    offs_c = pid_d * BDC + tl.arange(0, BDC)
    mask_t = offs_t < T
    mask_d = offs_d < D
    mask_c = offs_c < DC

    seq = tl.load(seq_idx + offs_t, mask=mask_t, other=0)
    start = tl.load(starts + seq, mask=mask_t, other=0)
    slot = tl.load(slots + seq, mask=mask_t, other=-1).to(tl.int64)
    has_state = (tl.load(initial + seq, mask=mask_t, other=0) != 0) & (slot >= 0)
    local = offs_t - start

    acc = tl.zeros((BT, BD), dtype=tl.float32)
    for k in tl.static_range(W):
        src = local + (k - H)           # position of this tap within the request's chunk
        in_chunk = src >= 0
        m_x = (mask_t & in_chunk)[:, None] & mask_d[None, :]
        xv = tl.load(x + (offs_t + (k - H)).to(tl.int64)[:, None] * D + offs_d[None, :],
                     mask=m_x, other=0.0).to(tl.float32)
        m_s = (mask_t & ~in_chunk & has_state)[:, None] & mask_d[None, :]
        sv = tl.load(state + slot[:, None] * s_slot + offs_d[None, :] * s_d + (src + H)[:, None] * s_h,
                     mask=m_s, other=0.0).to(tl.float32)
        wv = tl.load(w + offs_t.to(tl.int64)[:, None] * (W * DC) + k * DC + offs_c[None, :],
                     mask=mask_t[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        wf = tl.reshape(tl.broadcast_to(wv[:, :, None], (BT, BDC, G)), (BT, BD))
        acc += (xv + sv) * wf

    if HAS_NORM:
        grouped = tl.reshape(acc, (BT, BDC, G))
        scale = tl.maximum(tl.max(tl.abs(grouped), axis=2), 1.0)
        scaled = grouped / scale[:, :, None]
        ms = tl.sum(scaled * scaled, axis=2) / G
        denom = scale * tl.sqrt(ms + eps / (scale * scale))
        acc = tl.reshape(grouped / denom[:, :, None], (BT, BD))

    m_out = mask_t[:, None] & mask_d[None, :]
    out_offs = offs_t.to(tl.int64)[:, None] * D + offs_d[None, :]
    if HAS_GATE:
        gv = tl.load(g + out_offs, mask=m_out, other=0.0).to(tl.float32)
        acc = acc * (gv * tl.sigmoid(gv))
    tl.store(y + out_offs, acc.to(y.dtype.element_ty), mask=m_out)


@triton.jit
def _state_kernel(
    x, state, starts, slots, initial,
    s_slot, s_d, s_h,
    D: tl.constexpr, W: tl.constexpr, HP: tl.constexpr, BD: tl.constexpr,
):
    H: tl.constexpr = W - 1
    n, pid_d = tl.program_id(0), tl.program_id(1)
    slot = tl.load(slots + n).to(tl.int64)
    if slot < 0:
        return
    start = tl.load(starts + n)
    length = tl.load(starts + n + 1) - start
    has_state = tl.load(initial + n) != 0

    offs_d = pid_d * BD + tl.arange(0, BD)
    offs_h = tl.arange(0, HP)
    mask_d = offs_d < D
    mask_h = offs_h < H
    local = length - H + offs_h          # chunk position that becomes history slot h
    from_chunk = local >= 0
    new = tl.load(x + (start + local).to(tl.int64)[None, :] * D + offs_d[:, None],
                  mask=(from_chunk & mask_h)[None, :] & mask_d[:, None], other=0.0)
    base = state + slot * s_slot + offs_d[:, None] * s_d
    old = tl.load(base + (local + H)[None, :] * s_h,
                  mask=(~from_chunk & mask_h & has_state)[None, :] & mask_d[:, None], other=0.0)
    value = tl.where(from_chunk[None, :], new.to(state.dtype.element_ty), old)
    # Old history shifts within the same slot row: finish every read first.
    tl.debug_barrier()
    tl.store(base + offs_h[None, :] * s_h, value, mask=mask_h[None, :] & mask_d[:, None])


def supports(group_size: int, channels: int) -> bool:
    """The kernels tile channels by whole norm groups of a power-of-two size."""
    return group_size > 0 and group_size & (group_size - 1) == 0 and channels % group_size == 0


def _channel_block(group_size: int) -> int:
    return max(group_size, 128)


def packed_dynamic_conv_triton(x, weights, gate, state, starts, slots, has_initial,
                               group_size, norm=True, eps=1e-6, seq_idx=None, out=None):
    """Triton version of ``conv.packed_dynamic_conv`` (same arguments and result).

    x: [T, D]; weights: [T, W, D/G]; gate: [T, D] or None; state: [slots, D, W-1]
    (any strides). starts: [N+1]; slots, has_initial: [N]. seq_idx: [T] request
    index per token (computed when omitted). A negative slot means no history
    and no write-back. ``out`` ([T, D]) is filled in place when given.
    """
    tokens, channels = x.shape
    width = weights.shape[1]
    requests = slots.numel()
    if out is None:
        out = torch.empty_like(x)
    if tokens == 0:
        return out
    if not supports(group_size, channels):
        raise ValueError('Triton dynamic convolution needs a power-of-two group size dividing D')
    if weights.shape != (tokens, width, channels // group_size):
        raise ValueError('Incompatible dynamic-convolution weight/group dimensions')
    if state.shape[1:] != (channels, width - 1):
        raise ValueError('Convolution cache has the wrong shape')
    x, weights = x.contiguous(), weights.contiguous()
    gate = gate.contiguous() if gate is not None else None
    starts = starts.to(device=x.device, dtype=torch.int32)
    slots = slots.to(device=x.device, dtype=torch.int32).reshape(-1)
    has_initial = has_initial.to(device=x.device, dtype=torch.int8).reshape(-1)
    if seq_idx is None:
        seq_idx = torch.repeat_interleave(
            torch.arange(requests, device=x.device, dtype=torch.int32),
            starts[1:] - starts[:-1], output_size=tokens)
    block_d = _channel_block(group_size)
    block_t = 16  # one compiled variant for every batch size, so warmup covers it
    strides = state.stride()
    _conv_kernel[(triton.cdiv(tokens, block_t), triton.cdiv(channels, block_d))](
        x, weights, gate if gate is not None else x, out, state,
        starts, seq_idx, slots, has_initial,
        tokens, eps, *strides,
        D=channels, G=group_size, W=width, BT=block_t, BD=block_d,
        HAS_NORM=norm, HAS_GATE=gate is not None, num_warps=4,
    )
    _state_kernel[(requests, triton.cdiv(channels, block_d))](
        x, state, starts, slots, has_initial, *strides,
        D=channels, W=width, HP=triton.next_power_of_2(width - 1), BD=block_d, num_warps=4,
    )
    return out
