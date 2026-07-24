"""Offline validation of the per-body reference indexing (approach A).

The one thing that can silently break -- an env reading the WRONG body's
retargeted reference -- is pure index arithmetic, so it's testable without Isaac
Gym. This reproduces the intermimic.py logic exactly:
  - body-major expansion: object_id / references tiled [clips]*n_bodies
  - _to_body_block: data_id = _env_subject_idx * n_clips + clip
and asserts the invariants that guarantee correctness.

  python3 -m pytest tests/test_retarget_index.py -q
"""
import os

import torch


def _to_body_block(clip_ids, env_ids, env_subject_idx, n_clips, retarget=True):
    """Copy of IntermimicTask._to_body_block (no self)."""
    if not retarget:
        return clip_ids
    body = env_subject_idx[env_ids]
    mid = clip_ids + body * n_clips
    assert torch.all(mid // n_clips == body), "body-block invariant violated"
    return mid


def test_body_major_expansion_matches_offset():
    # 4 clips, 3 bodies. object_id_clip -> expanded body-major (as intermimic does
    # with object_id.repeat(n_b)).
    n_clips, subj = 4, ["sub2", "sub16", "sub10"]
    n_b = len(subj)
    object_id_clip = torch.tensor([0, 1, 0, 1])                 # clip -> object
    object_id_exp = object_id_clip.repeat(n_b)                  # body-major
    assert object_id_exp.shape[0] == n_b * n_clips

    # For every (body, clip): the expanded index must (a) land in the body block,
    # (b) recover the clip's object, (c) point at that body's retargeted file.
    for b in range(n_b):
        for c in range(n_clips):
            env_subject_idx = torch.tensor([b])
            data_id = _to_body_block(torch.tensor([c]), torch.tensor([0]),
                                     env_subject_idx, n_clips)
            m = int(data_id[0])
            assert m == b * n_clips + c
            assert m // n_clips == b                            # right body block
            assert m % n_clips == c                             # right clip within it
            assert int(object_id_exp[m]) == int(object_id_clip[c])   # object preserved
            # File the loader would read for this (body, clip):
            assert subj[m // n_clips] == subj[b]


def test_retarget_off_is_identity():
    # Retarget off -> data_id == clip, byte-identical to stock.
    n_clips = 5
    env_subject_idx = torch.tensor([0, 1, 2, 0])
    clip = torch.tensor([3, 1, 4, 0])
    out = _to_body_block(clip, torch.arange(4), env_subject_idx, n_clips, retarget=False)
    assert torch.equal(out, clip)


def test_wrong_body_would_be_caught():
    # A per-env vector where every env reads ITS body's block, never another's.
    n_clips, n_b = 10, 5
    env_subject_idx = torch.tensor([4, 2, 0, 3, 1, 4])         # arbitrary bodies
    env_ids = torch.arange(len(env_subject_idx))
    clips = torch.tensor([7, 0, 9, 3, 5, 1])
    data_id = _to_body_block(clips, env_ids, env_subject_idx, n_clips)
    # Every env's reference index decodes back to (its body, its clip).
    assert torch.equal(data_id // n_clips, env_subject_idx)
    assert torch.equal(data_id % n_clips, clips)


if __name__ == "__main__":
    test_body_major_expansion_matches_offset()
    test_retarget_off_is_identity()
    test_wrong_body_would_be_caught()
    print("all index invariants hold")
