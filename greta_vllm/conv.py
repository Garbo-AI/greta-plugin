"""Packed dynamic convolution with explicit, scheduler-owned request state.

Pure PyTorch correctness path. Works on CPU/CUDA, with no request-ID side cache.
State layout here is [cache slots, hidden channels, kernel_size - 1].
"""
import torch
import torch.nn.functional as F


def packed_dynamic_conv(x, weights, gate, state, starts, slots, has_initial,
                        group_size, norm=True, eps=1e-6):
    """Process packed request chunks and update only their selected cache slots.

    x: [T,D]; weights: [T,W,D/G]; gate: [T,D] or None.
    starts: [N+1] cumulative token offsets; slots/has_initial: [N].
    Inputs must describe nonempty disjoint requests, with distinct cache slots.
    """
    tokens, channels = x.shape
    width = weights.shape[1]
    history = width - 1
    requests = slots.numel()
    if tokens == 0:
        return x.clone()
    if channels % group_size or weights.shape != (tokens, width, channels // group_size):
        raise ValueError('Incompatible dynamic-convolution weight/group dimensions')
    if state.shape[1:] != (channels, history):
        raise ValueError('Convolution cache has the wrong shape')
    starts = starts.to(device=x.device, dtype=torch.long)
    slots = slots.to(device=x.device, dtype=torch.long).reshape(-1)
    has_initial = has_initial.to(device=x.device, dtype=torch.bool).reshape(-1)
    lengths = starts[1:] - starts[:-1]
    # output_size avoids the implicit device synchronization in repeat_interleave.
    seq = torch.repeat_interleave(torch.arange(requests, device=x.device),
                                 lengths, output_size=tokens)
    token_idx = torch.arange(tokens, device=x.device)
    local = token_idx - starts[seq]
    # A gather makes a snapshot, so writes at the end cannot corrupt this chunk.
    old = state.index_select(0, slots)
    old = torch.where(has_initial[:, None, None], old, torch.zeros_like(old))
    output = torch.zeros_like(x, dtype=torch.float32)
    for tap in range(width):
        offset = tap - history
        prior_local = local + offset
        current = x[(token_idx + offset).clamp(min=0)]
        if history:
            previous = old[seq, :, (prior_local + history).clamp(0, history - 1)]
            source = torch.where((prior_local >= 0)[:, None], current, previous)
        else:
            source = current
        tap_weight = weights[:, tap, :].repeat_interleave(group_size, dim=-1)
        output.add_(source.float() * tap_weight.float())
    if norm:
        grouped = output.reshape(tokens, channels // group_size, group_size)
        scale = grouped.abs().amax(-1, keepdim=True).clamp_min(1.0)
        scaled = grouped / scale
        denom = scale * torch.sqrt(scaled.square().mean(-1, keepdim=True) + eps / scale.square())
        output = (grouped / denom).reshape(tokens, channels)
    output = output.to(x.dtype)
    if gate is not None:
        output = output * F.silu(gate)
    if history:
        # Last W-1 tokens of old-history + this request's current chunk.
        tail_local = lengths[:, None] - history + torch.arange(history, device=x.device)
        absolute = starts[:-1, None] + tail_local
        new_tail = x[absolute.clamp(0, tokens - 1)].transpose(1, 2)
        old_tail = old.gather(2, (tail_local + history).clamp(0, history - 1)
                              [:, None, :].expand(-1, channels, -1))
        tail = torch.where((tail_local >= 0)[:, None, :], new_tail, old_tail)
        state.index_copy_(0, slots, tail.to(state.dtype))
    return output


def metadata_to_batch(metadata, positions):
    """Translate vLLM 0.30 SHORT_CONV metadata (decodes first) to packed chunks.

    Returns (starts, seq_idx, slots, initial, n): int32 cumulative offsets, the
    request index of each token (None when prefills are present; the Triton
    launcher derives it), int32 cache slots, int8 has-history flags and the
    number of real tokens. Uses device tensors and shapes only, so a decode-only
    step can be captured in a CUDA graph.
    """
    if metadata.num_accepted_tokens is not None:
        raise NotImplementedError('Greta speculative decoding is not supported')
    d, p = metadata.num_decode_tokens, metadata.num_prefill_tokens
    if d != metadata.num_decodes:
        raise NotImplementedError('Expected one token per decode request')
    starts = torch.arange(d + 1, device=positions.device, dtype=torch.int32)
    seq_idx = starts[:d]
    slots, initial = [], []
    if d:
        slots.append(metadata.state_indices_tensor_d.reshape(-1)[:d])
        # A one-token NEW prompt may be classified as a decode by vLLM.
        # Its position must reset a reused cache slot rather than read stale state.
        initial.append(positions[:d] > 0)
    if metadata.num_prefills:
        starts = torch.cat((starts, metadata.query_start_loc_p[1:].to(starts) + d))
        seq_idx = None
        slots.append(metadata.state_indices_tensor_p.reshape(-1)[:metadata.num_prefills])
        initial.append(metadata.has_initial_states_p[:metadata.num_prefills])
    if not slots:
        return starts, seq_idx, starts[:0], starts[:0].to(torch.int8), 0
    return (starts, seq_idx, torch.cat(slots).to(torch.int32),
            torch.cat(initial).to(torch.int8), d + p)
