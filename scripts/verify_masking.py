#!/usr/bin/env python3
"""Standalone verification that the per-env loss masking in
intermimic_agent_distill.py gives EXACT zero gradient contribution
from excluded envs.

Runs on CPU. No env, no Isaac Gym, no GPU needed. Pure PyTorch.

Mirrors the exact masking pattern used in the agent:
    masked_loss = (loss_per_env * valid_mask).sum() / valid_count

Verifies:
  1. Loss VALUE: excluded envs don't contribute (numerical check)
  2. Loss GRADIENT: w.r.t. mu, the rows for excluded envs are EXACTLY 0.0
  3. Backward pass through full agent-style total_loss (BC + actor + critic + bounds + entropy)
     produces zero gradient on every excluded env's predictions

Run:
    python scripts/verify_masking.py
"""
import torch
import torch.nn as nn


def per_env(x):
    """Reduce any trailing dims via mean so x is shape (B,). Matches agent."""
    return x.flatten(1).mean(-1) if x.dim() > 1 else x


def masked_mean(loss_raw, valid_mask, valid_count):
    """Same masked-mean recipe used in InterMimicAgentDistill.calc_gradients."""
    return (per_env(loss_raw) * valid_mask).sum() / valid_count


def main():
    torch.manual_seed(0)
    B = 16            # batch size (envs)
    A = 153           # action dim (matches policy)
    excluded = [3, 7, 11]   # indices we mark as excluded

    print(f"Setup: B={B} envs, action_dim={A}")
    print(f"Excluded env indices: {excluded}")

    # Fake policy: just a Linear layer obs -> (mu, value)
    # We give it requires_grad so gradients flow back.
    policy = nn.Linear(64, A + 1)  # last dim = value
    obs = torch.randn(B, 64)
    out = policy(obs)
    mu = out[:, :A]                 # (B, A) - policy mean action
    values = out[:, A:A+1]          # (B, 1) - critic value

    # Synthetic "expert" actions (teacher targets) and other PPO ingredients.
    expert_mus = torch.randn(B, A)
    advantage = torch.randn(B)
    old_log_probs = torch.randn(B)
    log_probs = (mu * 0.01).sum(-1)   # depends on mu so actor loss has grad
    return_batch = torch.randn(B, 1)
    sigma = torch.full((B, A), 0.1)

    # valid_mask: 1.0 for envs we keep, 0.0 for excluded.
    valid_mask = torch.ones(B)
    valid_mask[excluded] = 0.0
    valid_count = valid_mask.sum().clamp(min=1.0)
    print(f"valid_mask: {valid_mask.tolist()}")
    print(f"valid_count: {valid_count.item()}\n")

    # --- Compute the 5 loss terms exactly like the agent does ---
    # BC loss (per-env, summed over action dim, like _supervise_loss)
    e_loss_raw = ((mu - expert_mus.detach()) ** 2).sum(-1)        # (B,)

    # Actor loss
    ratio = torch.exp(old_log_probs - log_probs)
    surr1 = advantage * ratio
    surr2 = advantage * torch.clamp(ratio, 0.8, 1.2)
    a_loss_raw = torch.max(-surr1, -surr2)                        # (B,)

    # Critic loss
    c_loss_raw = (return_batch - values) ** 2                     # (B, 1)

    # Bounds loss
    mu_hi = torch.clamp_min(mu - 1.0, 0.0) ** 2
    mu_lo = torch.clamp_max(mu + 1.0, 0.0) ** 2
    b_loss_raw = (mu_hi + mu_lo).sum(-1)                          # (B,)

    # Entropy (depends on mu so we get nonzero grad — synthetic stand-in)
    entropy_raw = -(mu ** 2).mean(-1)                             # (B,)

    # --- Masked means ---
    e_loss = masked_mean(e_loss_raw, valid_mask, valid_count)
    a_loss = masked_mean(a_loss_raw, valid_mask, valid_count)
    c_loss = masked_mean(c_loss_raw, valid_mask, valid_count)
    b_loss = masked_mean(b_loss_raw, valid_mask, valid_count)
    entropy = masked_mean(entropy_raw, valid_mask, valid_count)

    total = e_loss + a_loss + 5.0 * c_loss + 10.0 * b_loss + 0.0 * entropy

    # --- The crucial test: does .backward() leak gradient to excluded envs? ---
    total.backward()
    mu_grad = policy.weight.grad   # gradient of total_loss w.r.t. policy params

    # Per-env gradient via re-forward with retain_graph to inspect
    # We check gradient w.r.t. each env's contribution to mu.
    # Cleanest way: re-run forward with mu having requires_grad and hook on mu.
    obs2 = obs.clone().detach()
    policy2 = nn.Linear(64, A + 1)
    policy2.load_state_dict(policy.state_dict())
    out2 = policy2(obs2)
    mu2 = out2[:, :A]
    values2 = out2[:, A:A+1]
    mu2.retain_grad()
    values2.retain_grad()

    e_loss_raw2 = ((mu2 - expert_mus.detach()) ** 2).sum(-1)
    log_probs2 = (mu2 * 0.01).sum(-1)
    ratio2 = torch.exp(old_log_probs - log_probs2)
    surr1_2 = advantage * ratio2
    surr2_2 = advantage * torch.clamp(ratio2, 0.8, 1.2)
    a_loss_raw2 = torch.max(-surr1_2, -surr2_2)
    c_loss_raw2 = (return_batch - values2) ** 2
    mu_hi2 = torch.clamp_min(mu2 - 1.0, 0.0) ** 2
    mu_lo2 = torch.clamp_max(mu2 + 1.0, 0.0) ** 2
    b_loss_raw2 = (mu_hi2 + mu_lo2).sum(-1)
    entropy_raw2 = -(mu2 ** 2).mean(-1)

    total2 = (masked_mean(e_loss_raw2, valid_mask, valid_count)
              + masked_mean(a_loss_raw2, valid_mask, valid_count)
              + 5.0 * masked_mean(c_loss_raw2, valid_mask, valid_count)
              + 10.0 * masked_mean(b_loss_raw2, valid_mask, valid_count)
              + 0.0 * masked_mean(entropy_raw2, valid_mask, valid_count))
    total2.backward()

    # mu2.grad shape: (B, A). For excluded envs, rows should be EXACTLY 0.
    print("=" * 60)
    print("GRADIENT CHECK — w.r.t. mu (policy mean action output)")
    print("=" * 60)
    excluded_grad = mu2.grad[excluded]
    included_idx = [i for i in range(B) if i not in excluded]
    included_grad = mu2.grad[included_idx]

    excl_max = excluded_grad.abs().max().item()
    incl_max = included_grad.abs().max().item()
    print(f"  excluded rows ({len(excluded)} envs)  | max |grad| = {excl_max:.3e}   "
          f"<-- MUST be 0.0")
    print(f"  included rows ({len(included_idx)} envs) | max |grad| = {incl_max:.3e}   "
          f"<-- should be nonzero")

    assert excl_max == 0.0, (
        f"FAIL: excluded envs received nonzero grad ({excl_max:.3e}). "
        f"Masking is leaking gradient."
    )
    assert incl_max > 0.0, (
        f"FAIL: included envs received zero grad — they should be learning."
    )

    print("\n" + "=" * 60)
    print("GRADIENT CHECK — w.r.t. values (critic output)")
    print("=" * 60)
    excl_v = values2.grad[excluded]
    incl_v = values2.grad[included_idx]
    excl_v_max = excl_v.abs().max().item()
    incl_v_max = incl_v.abs().max().item()
    print(f"  excluded rows | max |grad| = {excl_v_max:.3e}   <-- MUST be 0.0")
    print(f"  included rows | max |grad| = {incl_v_max:.3e}   <-- should be nonzero")
    assert excl_v_max == 0.0, f"FAIL: critic gradient leaked to excluded envs ({excl_v_max:.3e})"

    print("\n" + "=" * 60)
    print("LOSS VALUE CHECK — exclude vs include same envs")
    print("=" * 60)
    # Sanity: if we re-mask the SAME envs into the loss (mask=1 for them too),
    # the total loss should differ. This proves masking is doing work.
    valid_mask_all_ones = torch.ones_like(valid_mask)
    e_loss_unmasked = masked_mean(e_loss_raw.detach(), valid_mask_all_ones, valid_mask_all_ones.sum())
    e_loss_masked = masked_mean(e_loss_raw.detach(), valid_mask, valid_count)
    print(f"  BC loss WITH    excluded envs (all envs counted) = {e_loss_unmasked.item():.4f}")
    print(f"  BC loss WITHOUT excluded envs (mask applied)     = {e_loss_masked.item():.4f}")
    print(f"  Difference (proof mask is changing the loss)     = "
          f"{abs(e_loss_unmasked.item() - e_loss_masked.item()):.4e}")

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED — masking gives EXACT zero gradient leak.")
    print("=" * 60)


if __name__ == "__main__":
    main()
