#!/usr/bin/env python3
"""PSI reference-buffer update, without the per-motion Python loop.

PSI (Physically-corrected State Initialization) keeps, per motion and per timestep,
a few candidate reference states that the policy ACTUALLY reached in simulation, and
initialises episodes from those instead of from mocap frames the body cannot hold.
After each batch of resets the best-scoring rollout segments overwrite the weakest
slot in that buffer.

The original implementation allocated `(n_reset, num_motions, T)` and then walked
`for i in range(num_motions): for j in range(T)` on every reset -- costs that scale
with the SIZE OF THE DATASET rather than with how much actually reset. That was
tolerable at 293 motions; body-major retargeting takes it to 2704 (52 clips x 52
bodies), which is 1.9 GB allocated and ~1M Python iterations per reset batch, and
measured out at ~3x slower steps end to end.

Only the motions that appear in this batch's `data_id` can be written: every other
motion's `adjust_reward` row is all zeros, so the `> 0.5` guard rejects it. This
version therefore works over `data_id.unique()` and vectorises the (motion, time)
loop. The writes are independent -- the slot chosen at (motion, j) is read from
column j only and written to column j only -- so there is no ordering to preserve.

Semantics are intended to be BIT-IDENTICAL to the original; tests/test_psi_update.py
pins that against a verbatim transcription of the old loop.
"""
import torch


def psi_buffer_update(data_id, start_index, end_index, end_i, state,
                      ref_reward, hoi_refs, max_episode_length, rollout_length):
    """Fold this batch's rollouts into the PSI buffers, in place.

    Args:
        data_id:        (n_reset,) motion index each resetting env was playing
        start_index:    (n_reset,) episode start frame
        end_index:      (n_reset,) frame the episode actually reached
        end_i:          (n_reset,) min(motion length, rollout_length + start)
        state:          (n_reset, >=rollout_length, F) states achieved this rollout
        ref_reward:     (num_motions, psi, T) buffer scores -- MODIFIED IN PLACE
        hoi_refs:       (num_motions, psi, T, F) buffer states -- MODIFIED IN PLACE
                        (may live on a different device than the rest, see cpuMotionData)
        max_episode_length: (num_motions,)
        rollout_length: int

    Returns:
        int -- how many (motion, timestep) slots were written, for logging/tests.
    """
    dev = state.device
    n_reset = int(data_id.shape[0])
    T = int(ref_reward.shape[2])
    if n_reset == 0:
        return 0

    j = torch.arange(T, device=dev)

    # --- per-reset score ramp -------------------------------------------------
    # Original: for each reset i with end > start+30, fill columns
    # [start+10, end-10) with (end_index - t) / (end_i - t). Everything else 0.
    # Kept as an (n_reset, T) plane instead of an (n_reset, num_motions, T) volume
    # `state` is gathered with reset-row indices in [0, n_reset), so it must have
    # one row PER RESET, aligned elementwise with data_id/start_index/end_index.
    # Passing a narrowed `state` gathers off the end and surfaces only as a
    # device-side assert in IndexKernel.cu, thousands of epochs in, with a
    # traceback that points here but says nothing about why. Fail readably.
    if state.shape[0] != n_reset:
        raise ValueError(
            f"state has {state.shape[0]} rows but data_id has {n_reset}: these must "
            f"be indexed with the SAME reset mask, or every row lookup is misaligned "
            f"(and out of bounds when state is the shorter one).")

    # -- the motion axis was one-hot on data_id[i], so it stored nothing.
    long_enough = (end_index > start_index + 30)
    in_window = ((j[None, :] >= (start_index + 10)[:, None])
                 & (j[None, :] < (end_index - 10)[:, None])
                 & long_enough[:, None])
    ramp = (end_index[:, None] - j[None, :]) / (end_i[:, None] - j[None, :])
    ramp = torch.where(in_window, ramp.to(ref_reward.dtype),
                       torch.zeros((), dtype=ref_reward.dtype, device=dev))

    # --- max over the resets that share a motion ------------------------------
    motions, inv = torch.unique(data_id, return_inverse=True)     # (M,), (n_reset,)
    M = int(motions.shape[0])
    scatter_idx = inv[:, None].expand(n_reset, T)

    adjust = torch.zeros((M, T), dtype=ref_reward.dtype, device=dev)
    adjust.scatter_reduce_(0, scatter_idx, ramp, reduce="amax", include_self=True)

    # argmax to go with it. `amax` returns one of its inputs bitwise, so comparing
    # for equality is exact; `amin` over the matching row indices then reproduces
    # torch.max's first-occurrence tie-break deterministically.
    rows = torch.arange(n_reset, device=dev)[:, None].expand(n_reset, T)
    cand = torch.where(ramp == adjust[inv], rows, torch.full_like(rows, n_reset))
    winner = torch.full((M, T), n_reset, dtype=rows.dtype, device=dev)
    winner.scatter_reduce_(0, scatter_idx, cand, reduce="amin", include_self=True)
    # Motions nothing was written for keep adjust == 0 and so fail the > 0.5 guard
    # below; 0 just matches what argmax over an all-zero column used to return.
    winner = torch.where(winner >= n_reset, torch.zeros_like(winner), winner)

    # --- which (motion, timestep) slots are eligible --------------------------
    mel = max_episode_length[motions].to(dev)
    # Original skipped a motion when mel < rollout_length, and broke out of the
    # time loop at the first j with mel - j < rollout_length; since j ascends,
    # that is exactly "keep j while mel - j >= rollout_length".
    eligible = ((mel >= rollout_length)[:, None]
                & ((mel[:, None] - j[None, :]) >= rollout_length))
    offset = j[None, :] - start_index[winner]                     # (M, T)
    cond = (eligible & (offset > 0) & (offset < rollout_length) & (adjust > 0.5))
    if not bool(cond.any()):
        return 0

    # --- weakest slot per (motion, timestep), then scatter the winners in -----
    # Slot 0 is the mocap reference and is never overwritten, hence the 1: view.
    slot = ref_reward[motions][:, 1:, :].argmin(dim=1) + 1        # (M, T)

    m_sel, j_sel = cond.nonzero(as_tuple=True)
    g_sel = motions[m_sel]                                        # global motion ids
    s_sel = slot[m_sel, j_sel]
    ref_reward[g_sel, s_sel, j_sel] = adjust[m_sel, j_sel].to(ref_reward.dtype)
    hoi_refs[g_sel.to(hoi_refs.device), s_sel.to(hoi_refs.device), j_sel.to(hoi_refs.device)] = \
        state[winner[m_sel, j_sel], offset[m_sel, j_sel]].to(hoi_refs.device)
    return int(m_sel.shape[0])
