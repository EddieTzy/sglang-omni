# SPDX-License-Identifier: Apache-2.0
"""Integer-only Fish decode feedback update, with sampling kept in the model."""

import torch
import triton
import triton.language as tl


@triton.jit
def advance_decode_state_kernel(
    CODES,
    HISTORY,
    COUNT,
    STEP,
    VQ,
    N,
    CODE_STRIDE: tl.constexpr,
    HISTORY_STRIDE: tl.constexpr,
    VQ_STRIDE: tl.constexpr,
    W: tl.constexpr,
    K: tl.constexpr,
    EOS: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    col = tl.arange(0, COLS)
    live = row < N
    token = tl.load(CODES + row * CODE_STRIDE, live, other=EOS)
    count = tl.load(COUNT + row, live, other=0)
    step = tl.load(STEP + row, live, other=0)
    active = live & (token != EOS)
    shift = active & (count >= W)
    source = tl.minimum(col[None, :] + shift[:, None].to(tl.int32), W - 1)
    history = tl.load(
        HISTORY + row[:, None] * HISTORY_STRIDE + source,
        live[:, None] & (col[None, :] < W),
        other=0,
    )
    history = tl.where(
        active[:, None] & (col[None, :] == tl.minimum(count, W - 1)[:, None]),
        token[:, None],
        history,
    )
    codes = tl.load(
        CODES + row[:, None] * CODE_STRIDE + col[None, :] + 1,
        live[:, None] & (col[None, :] < K),
        other=0,
    )
    # Each program owns disjoint rows. Complete all old-history reads before
    # writing the shifted history, including windows spanning multiple warps.
    tl.debug_barrier()
    tl.store(
        HISTORY + row[:, None] * HISTORY_STRIDE + col[None, :],
        history,
        live[:, None] & (col[None, :] < W),
    )
    tl.store(COUNT + row, tl.where(active, tl.minimum(count + 1, W), count), live)
    tl.store(STEP + row, step + active.to(step.dtype), live)
    # EOS still updates VQ feedback; only history and sampling step freeze.
    tl.store(
        VQ + row[:, None] * VQ_STRIDE + col[None, :],
        codes,
        live[:, None] & (col[None, :] < K),
    )


def advance_decode_state(
    output_codes: torch.Tensor,
    prev_tokens: torch.Tensor,
    prev_token_count: torch.Tensor,
    step_count: torch.Tensor,
    vq_codes: torch.Tensor,
    *,
    im_end_token_id: int,
) -> None:
    """Update the first output_codes.shape[0] rows in place without host sync.

    Inputs are the persistent Fish buffers with contiguous columns; row strides
    may include capacity padding. Output codes contain [semantic, cb0, ...].
    EOS copies codebooks but retains repetition history/count and sampling step.
    """
    n = output_codes.shape[0]
    w = prev_tokens.shape[1]
    k = vq_codes.shape[1]
    if n:
        advance_decode_state_kernel[(triton.cdiv(n, 4),)](
            output_codes,
            prev_tokens,
            prev_token_count,
            step_count,
            vq_codes,
            n,
            output_codes.stride(0),
            prev_tokens.stride(0),
            vq_codes.stride(0),
            w,
            k,
            im_end_token_id,
            ROWS=4,
            COLS=triton.next_power_of_2(max(w, k)),
            num_warps=4,
        )
    else:
        pass
