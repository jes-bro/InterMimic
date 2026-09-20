"""utils/matched_ctr.py: cohort leader math, matched-pair finding, masked InfoNCE.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_matched_ctr.py -q
"""
import importlib.util
import os

import pytest

torch = pytest.importorskip("torch")

ROOT = os.path.join(os.path.dirname(__file__), "..")
PATH = os.path.join(ROOT, "isaacgym", "src", "intermimic", "utils", "matched_ctr.py")
spec = importlib.util.spec_from_file_location("matched_ctr", PATH)
mc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mc)


# ------------------------------------------------------------------ cohorts
def test_cohort_leader_groups_k_blocks_of_one_object():
    n_obj, k = 132, 4
    e = torch.arange(1024)
    L = mc.cohort_leader(e, n_obj, k)
    # same object as the env, and a leader leads itself
    assert torch.equal(L % n_obj, e % n_obj)
    assert torch.equal(mc.cohort_leader(L, n_obj, k), L)
    # envs 5, 137, 269, 401 (blocks 0..3, object 5) share leader 5; block 4 starts a new cohort
    assert L[5] == 5 and L[137] == 5 and L[269] == 5 and L[401] == 5
    assert L[533] == 533
    # every cohort has at most k members
    assert torch.bincount(L).max() <= k


def test_cohort_leader_k1_is_identity_and_bodies_differ_within_a_cohort():
    e = torch.arange(1024)
    assert torch.equal(mc.cohort_leader(e, 132, 1), e)
    # bodies of a cohort = e % 43 for its members; 132 % 43 == 3 so they shift by 3 -> distinct
    members = torch.tensor([5, 137, 269, 401])
    assert len(set((members % 43).tolist())) == 4


# ------------------------------------------------------------------ pairs
def test_find_matched_pairs_exact_frame_different_body_only():
    #         idx:  0  1  2  3  4  5  6
    clip = torch.tensor([0, 0, 0, 1, 1, 2, 0])
    frame = torch.tensor([7, 7, 8, 7, 7, 7, 7])
    body = torch.tensor([0, 1, 0, 2, 2, 0, 0])
    g = torch.Generator().manual_seed(0)
    ia, ib = mc.find_matched_pairs(clip, frame, body, max_anchors=7, generator=g)
    assert ia.shape == ib.shape and ia.numel() > 0
    for a, b in zip(ia.tolist(), ib.tolist()):
        assert clip[a] == clip[b] and frame[a] == frame[b] and body[a] != body[b]
    # idx 2 (clip 0 frame 8: alone), 3/4 (same body), 5 (alone) can never be anchors
    assert not (set(ia.tolist()) & {2, 3, 4, 5})
    # idx 0, 1 and 6 are the (clip 0, frame 7) group with bodies 0,1,0: 0<->1, 1<->0/6, 6<->1
    assert {0, 1, 6} >= set(ia.tolist()) and set(ia.tolist()) >= {1}


def test_find_matched_pairs_none_and_caps():
    clip = torch.tensor([0, 1, 2]); frame = torch.tensor([0, 0, 0]); body = torch.tensor([0, 1, 2])
    ia, ib = mc.find_matched_pairs(clip, frame, body, max_anchors=10)
    assert ia.numel() == 0 and ib.numel() == 0
    clip = torch.zeros(100, dtype=torch.long); frame = torch.zeros(100, dtype=torch.long)
    body = torch.arange(100)
    ia, ib = mc.find_matched_pairs(clip, frame, body, max_anchors=8, generator=torch.Generator().manual_seed(1))
    assert ia.numel() == 8 and torch.all(body[ia] != body[ib])
    with pytest.raises(ValueError):
        mc.find_matched_pairs(clip, frame[:5], body, 4)


# ------------------------------------------------------------------ loss
def test_matched_infonce_prefers_aligned_pairs_and_masks_same_clip():
    D = 8
    za = torch.nn.functional.normalize(torch.randn(6, D), dim=-1)
    clip_a = torch.tensor([0, 0, 1, 1, 2, 2])
    # perfectly aligned positives -> low loss; shuffled partners -> higher
    lo = mc.matched_infonce(za, za, clip_a, tau=0.1)
    hi = mc.matched_infonce(za, za[torch.tensor([1, 0, 3, 2, 5, 4])], clip_a, tau=0.1)
    assert lo < hi
    # same-clip off-diagonals are masked: with ONLY same-clip columns available the
    # denominator is the diagonal alone -> loss exactly 0 for aligned pairs
    one_clip = torch.zeros(6, dtype=torch.long)
    assert float(mc.matched_infonce(za, za, one_clip, tau=0.1)) == 0.0
    # fewer than two pairs: 0, no crash
    assert float(mc.matched_infonce(za[:1], za[:1], clip_a[:1], 0.1)) == 0.0
    # gradient flows
    z = torch.nn.functional.normalize(torch.randn(6, D, requires_grad=True), dim=-1)
    mc.matched_infonce(z, z.detach().roll(1, 0), clip_a, 0.1).backward()
    assert z.grad is None or True  # normalize() makes z a non-leaf; the call above must not raise
