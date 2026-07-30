#!/usr/bin/env python3
"""psi_buffer_update must be bit-identical to the loop it replaces.

The replacement exists purely for speed (the original allocated
(n_reset, num_motions, T) and looped over every motion x timestep on every reset,
which body-major retargeting blew up from 293 motions to 2704). A speedup that
quietly changes what lands in the PSI buffer would invalidate every run that uses
it, so the whole suite is differential: run both implementations on the same random
inputs and require the buffers to come out equal.

`_reference_update` below is a verbatim transcription of intermimic.py's pre-fix
loop. Do not "clean it up" -- its value is being the thing we no longer trust
ourselves to reason about.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_psi_update.py -v
"""
import importlib.util
import os

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "psi_update", os.path.join(REPO, "isaacgym/src/intermimic/utils/psi_update.py"))
psi_update = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(psi_update)
psi_buffer_update = psi_update.psi_buffer_update


def _reference_update(data_id, start_index, end_index, end_i, state,
                      ref_reward, hoi_refs, max_episode_length, rollout_length):
    """Verbatim transcription of the original intermimic.py:1506-1534 loop."""
    n_reset = data_id.shape[0]
    reward = torch.zeros((n_reset, ref_reward.shape[0], ref_reward.shape[2]),
                         device=state.device, dtype=ref_reward.dtype)
    for i in range(n_reset):
        if end_index[i] > start_index[i] + 30:
            index_tensor = torch.arange(start_index[i] + 10, end_index[i] - 10,
                                        device=start_index.device)
            reward[i, data_id[i], start_index[i] + 10:end_index[i] - 10] = \
                ((end_index[i] - index_tensor) / (end_i[i] - index_tensor)).to(ref_reward.dtype)

    adjust_reward, adjust_reward_index = reward.max(dim=0)
    written = 0
    for i in range(reward.shape[1]):
        if max_episode_length[i] < rollout_length:
            continue
        for j in range(reward.shape[2]):
            if max_episode_length[i] - j < rollout_length:
                break
            value, index = ref_reward[i, 1:, j].min(dim=0)
            index = index + 1
            id1 = adjust_reward_index[i, j]
            idx = j - start_index[adjust_reward_index[i, j]]

            if idx > 0 and idx < rollout_length and adjust_reward[i, j] > 0.5:
                ref_reward[i, index, j] = adjust_reward[i, j]
                hoi_refs[i, index, j] = state[id1, idx].to(hoi_refs.device)
                written += 1
    return written


def _make_case(seed, num_motions, n_reset, T=120, psi=3, F=5, rollout_length=40,
               short_motions=0):
    """Random but structurally valid inputs. `short_motions` marks some motions as
    too short to be eligible, exercising the skip/break branches.

    The constants are not arbitrary. A rollout must satisfy end > start + 30 to fill
    anything at all, while end < end_i = min(motion_len, rollout_length + start)
    bounds it -- so rollout_length has to exceed 31 for any episode to qualify. With
    rollout_length=40 the ramp at the window start is (d-10)/30 for d in (30, 40),
    i.e. 0.67..1.0, which clears the > 0.5 write guard. Get this wrong and every
    equivalence test passes while comparing two no-ops (see
    test_it_actually_writes_something).
    """
    g = torch.Generator().manual_seed(seed)
    data_id = torch.randint(0, num_motions, (n_reset,), generator=g)
    start_index = torch.randint(0, 20, (n_reset,), generator=g)
    duration = torch.randint(31, rollout_length, (n_reset,), generator=g)
    end_index = start_index + duration
    max_episode_length = torch.full((num_motions,), T, dtype=torch.long)
    if short_motions:
        max_episode_length[:short_motions] = rollout_length - 1
    mel_reset = max_episode_length[data_id]
    end_i = torch.minimum(mel_reset, rollout_length + start_index)
    end_index = torch.minimum(end_index, end_i - 1)          # the original asserts this
    state = torch.randn(n_reset, rollout_length, F, generator=g)
    ref_reward = torch.rand(num_motions, psi, T, generator=g)
    hoi_refs = torch.randn(num_motions, psi, T, F, generator=g)
    return dict(data_id=data_id, start_index=start_index, end_index=end_index,
                end_i=end_i, state=state, ref_reward=ref_reward, hoi_refs=hoi_refs,
                max_episode_length=max_episode_length, rollout_length=rollout_length)


def _run_both(case):
    a = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in case.items()}
    b = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in case.items()}
    n_ref = _reference_update(**a)
    n_new = psi_buffer_update(**b)
    return (a["ref_reward"], a["hoi_refs"], n_ref), (b["ref_reward"], b["hoi_refs"], n_new)


@pytest.mark.parametrize("seed,num_motions,n_reset", [
    (0, 8, 32),        # many resets per motion -- ties and contention
    (1, 64, 32),       # sparse: most motions untouched, the retargeting regime
    (2, 200, 16),      # very sparse
    (3, 4, 64),        # dense
    (7, 33, 33),       # equal-ish
])
def test_matches_reference(seed, num_motions, n_reset):
    (rr_a, hr_a, n_a), (rr_b, hr_b, n_b) = _run_both(_make_case(seed, num_motions, n_reset))
    assert n_a == n_b, f"wrote {n_b} slots, reference wrote {n_a}"
    assert torch.equal(rr_a, rr_b), \
        f"ref_reward differs, max |d| = {(rr_a - rr_b).abs().max():.3e}"
    assert torch.equal(hr_a, hr_b), \
        f"hoi_refs differs, max |d| = {(hr_a - hr_b).abs().max():.3e}"


def test_matches_reference_with_short_motions():
    """Motions shorter than rollout_length hit the `continue`/`break` branches."""
    case = _make_case(11, num_motions=20, n_reset=24, short_motions=6)
    (rr_a, hr_a, n_a), (rr_b, hr_b, n_b) = _run_both(case)
    assert n_a == n_b and torch.equal(rr_a, rr_b) and torch.equal(hr_a, hr_b)


def test_it_actually_writes_something():
    """Guard against a vacuous pass: the cases above must exercise real writes."""
    _, (_, _, n_new) = _run_both(_make_case(0, 8, 32))
    assert n_new > 0, "no slots written -- the equivalence tests would prove nothing"


def test_untouched_motions_are_left_alone():
    """A motion with no resets this batch must come back byte-identical."""
    case = _make_case(5, num_motions=40, n_reset=8)
    before = case["ref_reward"].clone(), case["hoi_refs"].clone()
    absent = sorted(set(range(40)) - set(case["data_id"].tolist()))
    assert absent, "fixture produced no absent motions"
    psi_buffer_update(**case)
    for m in absent:
        assert torch.equal(case["ref_reward"][m], before[0][m]), f"motion {m} touched"
        assert torch.equal(case["hoi_refs"][m], before[1][m]), f"motion {m} touched"


def test_slot_zero_is_never_overwritten():
    """Slot 0 holds the original mocap reference; PSI only replaces slots 1..psi-1."""
    case = _make_case(3, num_motions=6, n_reset=48)
    before = case["ref_reward"][:, 0].clone(), case["hoi_refs"][:, 0].clone()
    assert psi_buffer_update(**case) > 0
    assert torch.equal(case["ref_reward"][:, 0], before[0])
    assert torch.equal(case["hoi_refs"][:, 0], before[1])


def test_empty_reset_batch_is_a_noop():
    case = _make_case(0, num_motions=8, n_reset=32)
    for k in ("data_id", "start_index", "end_index", "end_i"):
        case[k] = case[k][:0]
    case["state"] = case["state"][:0]
    before = case["ref_reward"].clone()
    assert psi_buffer_update(**case) == 0
    assert torch.equal(case["ref_reward"], before)


def test_hoi_refs_on_a_different_device_still_works():
    """cpuMotionData keeps hoi_refs on CPU while the rest is on GPU; the CPU-only
    stand-in here at least pins that the code does not assume one shared device."""
    case = _make_case(0, num_motions=8, n_reset=32)
    case["hoi_refs"] = case["hoi_refs"].to("cpu")
    assert psi_buffer_update(**case) > 0


def test_misaligned_state_fails_readably():
    """A narrowed `state` must raise, not gather off the end of the tensor.

    This is what crashed both teacher runs at ~epoch 1268: the caller built
    `state` from a narrowed reset mask while data_id/start_index/end_index came
    from the original one, so `winner` (in [0, n_reset)) indexed past the end of
    `state`. On CUDA that is an opaque device-side assert in IndexKernel.cu.
    """
    import pytest
    n_reset, T, F, n_motions, slots = 6, 40, 5, 3, 3
    data_id = torch.tensor([0, 1, 2, 0, 1, 2])
    start_index = torch.zeros(n_reset, dtype=torch.long)
    end_index = torch.full((n_reset,), T - 1, dtype=torch.long)
    end_i = torch.full((n_reset,), T, dtype=torch.long)
    ref_reward = torch.ones((n_motions, slots, T))
    hoi_refs = torch.zeros((n_motions, slots, T, F))
    mel = torch.full((n_motions,), T, dtype=torch.long)

    # state with one row per reset: fine (may or may not write, must not raise)
    psi_buffer_update(data_id, start_index, end_index, end_i,
                      torch.randn(n_reset, T, F), ref_reward, hoi_refs, mel, 10)

    # state narrowed to fewer rows: must raise, and name the mismatch
    with pytest.raises(ValueError, match="same reset mask|rows but data_id"):
        psi_buffer_update(data_id, start_index, end_index, end_i,
                          torch.randn(n_reset - 2, T, F), ref_reward, hoi_refs, mel, 10)
