#!/usr/bin/env python3
"""RaggedMotion must answer every index the padded tensors answered, identically.

The ragged store exists purely for memory (3.4x on the 13-source OMOMO teacher).
A store that quietly returned a different frame than the padded tensor would
change what the policy tracks and what PSI initialises from, invalidating the
arm that uses it. So the suite is DIFFERENTIAL: build the padded tensors exactly
the way intermimic.py's `_load_motion` does (F.pad to the longest clip, stack,
repeat over PSI slots) and a RaggedMotion from the SAME clips, then require equal
results for every read the task performs, the PSI write, and a full
psi_buffer_update on both.

Three facts make the store correct, and each has a test here:
  1. offsets is an exclusive prefix sum of lengths          (test_offsets_prefix_sum)
  2. write_clip and __getitem__ use the same row formula    (every gather test)
  3. no reader asks for t >= lengths[m] -- enforced on CPU  (test_out_of_range_*)
     and checked once on GPU with RAGGED_CHECK_BOUNDS=1 through a rollout.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_ragged_motion.py -v
"""
import importlib.util
import os

import pytest
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ragged_motion = _load("ragged_motion", "isaacgym/src/intermimic/utils/ragged_motion.py")
psi_update = _load("psi_update", "isaacgym/src/intermimic/utils/psi_update.py")
RaggedMotion = ragged_motion.RaggedMotion
scan_clip_lengths = ragged_motion.scan_clip_lengths
psi_buffer_update = psi_update.psi_buffer_update

DATA_W, REFS_W = 23, 11          # small stand-ins for 1211 / 332
TOPK = 3


def make_clips(lengths, width, seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(n, width, generator=g) for n in lengths]


def padded_like_loader(clips, topk=None):
    """Verbatim shape of intermimic.py _load_motion's padded assembly."""
    max_length = max(c.shape[0] for c in clips)
    out = [F.pad(c, (0, 0, 0, max_length - c.shape[0]), "constant", 0) for c in clips]
    t = torch.stack(out, dim=0)
    if topk is not None:
        t = t.unsqueeze(1).repeat(1, topk, 1, 1)
    return t


def ragged_from(clips, topk=None, device="cpu"):
    r = RaggedMotion([c.shape[0] for c in clips], topk=topk, device=device)
    for i, c in enumerate(clips):
        r.write_clip(i, c)
    return r.assert_complete()


@pytest.fixture
def lengths():
    # Deliberately ragged: a 1-frame clip, a long outlier, duplicates.
    return [7, 1, 30, 12, 12, 5, 19]


@pytest.fixture
def stores(lengths):
    data_clips = make_clips(lengths, DATA_W, seed=1)
    refs_clips = make_clips(lengths, REFS_W, seed=2)
    return dict(
        lengths=lengths,
        pad_data=padded_like_loader(data_clips),
        pad_refs=padded_like_loader(refs_clips, topk=TOPK),
        rag_data=ragged_from(data_clips),
        rag_refs=ragged_from(refs_clips, topk=TOPK),
    )


def valid_index(lengths, n, seed, topk=None):
    """Random (m, t[, s]) with t < lengths[m], the only reads the task performs."""
    g = torch.Generator().manual_seed(seed)
    L = torch.as_tensor(lengths)
    m = torch.randint(0, len(lengths), (n,), generator=g)
    t = (torch.rand(n, generator=g) * L[m]).long()
    assert bool((t < L[m]).all())
    if topk is None:
        return m, t
    s = torch.randint(0, topk, (n,), generator=g)
    return m, s, t


# ---- fact 1: offsets --------------------------------------------------------
def test_offsets_prefix_sum(lengths):
    r = RaggedMotion(lengths)
    assert int(r.offsets[0]) == 0
    for i in range(len(lengths) - 1):
        assert int(r.offsets[i + 1]) == int(r.offsets[i]) + lengths[i]
    assert r.total_frames == sum(lengths)
    assert r.max_length == max(lengths)


# ---- fact 2: every read the task performs -----------------------------------
def test_gather_frame(stores):
    m, t = valid_index(stores["lengths"], 500, seed=3)
    torch.testing.assert_close(stores["rag_data"][(m, t)], stores["pad_data"][m, t],
                               rtol=0, atol=0)


def test_gather_frame_columns(stores):
    """extract_data_component: (data_id, t, slice(start, end))."""
    m, t = valid_index(stores["lengths"], 500, seed=4)
    for cols in (slice(0, 3), slice(3, 7), slice(9, DATA_W), slice(None)):
        torch.testing.assert_close(stores["rag_data"][(m, t, cols)],
                                   stores["pad_data"][m, t, cols], rtol=0, atol=0)


def test_gather_refs_slot(stores):
    """extract_ref_component: (data_id, ref_index, t, slice)."""
    m, s, t = valid_index(stores["lengths"], 500, seed=5, topk=TOPK)
    torch.testing.assert_close(stores["rag_refs"][(m, s, t)], stores["pad_refs"][m, s, t],
                               rtol=0, atol=0)
    for cols in (slice(0, 3), slice(4, 9)):
        torch.testing.assert_close(stores["rag_refs"][(m, s, t, cols)],
                                   stores["pad_refs"][m, s, t, cols], rtol=0, atol=0)


def test_every_valid_cell_matches(stores):
    """Exhaustive, not sampled: every (m, t) below the clip length, every slot."""
    for m, n in enumerate(stores["lengths"]):
        t = torch.arange(n)
        mm = torch.full((n,), m)
        torch.testing.assert_close(stores["rag_data"][(mm, t)], stores["pad_data"][mm, t],
                                   rtol=0, atol=0)
        for s in range(TOPK):
            ss = torch.full((n,), s)
            torch.testing.assert_close(stores["rag_refs"][(mm, ss, t)],
                                       stores["pad_refs"][mm, ss, t], rtol=0, atol=0)


def test_scalar_index(stores):
    """play_dataset_step-style Python ints are accepted too."""
    torch.testing.assert_close(stores["rag_data"][(2, 29)], stores["pad_data"][2, 29],
                               rtol=0, atol=0)


def test_indices_on_other_device_are_moved(stores):
    """Readers hand GPU index tensors to a CPU store under cpuMotionData; the
    store must not care which device the indices arrive on."""
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    m, t = valid_index(stores["lengths"], 64, seed=6)
    out = stores["rag_data"][(m.cuda(), t.cuda())]
    assert out.device.type == "cpu"
    torch.testing.assert_close(out, stores["pad_data"][m, t], rtol=0, atol=0)


# ---- the PSI write, both as a bare setitem and through psi_buffer_update ----
def test_setitem_matches_padded(stores):
    m, s, t = valid_index(stores["lengths"], 200, seed=7, topk=TOPK)
    val = torch.randn(200, REFS_W)
    stores["pad_refs"][m, s, t] = val
    stores["rag_refs"][(m, s, t)] = val
    mm, ss, tt = valid_index(stores["lengths"], 2000, seed=8, topk=TOPK)
    torch.testing.assert_close(stores["rag_refs"][(mm, ss, tt)],
                               stores["pad_refs"][mm, ss, tt], rtol=0, atol=0)


PSI_LENGTHS = [100, 1, 64, 40, 40, 5, 80]     # long enough for the ramp to fire
PSI_ROLLOUT = 50


def _psi_stores():
    clips = make_clips(PSI_LENGTHS, REFS_W, seed=2)
    return padded_like_loader(clips, topk=TOPK), ragged_from(clips, topk=TOPK)


def _psi_batch(seed, n_reset=60):
    """Resets shaped like the task's: start in [0, mel - rollout), the episode
    ends before start + rollout (compute_hoi_reset), and long enough
    (end > start + 30) that psi_update's ramp is nonzero for most of them."""
    mel = torch.as_tensor(PSI_LENGTHS)
    g = torch.Generator().manual_seed(seed)
    data_id = torch.randint(0, len(PSI_LENGTHS), (n_reset,), generator=g)
    start = (torch.rand(n_reset, generator=g) * (mel[data_id] - PSI_ROLLOUT).clamp(min=1)).long()
    end_i = torch.minimum(mel[data_id], PSI_ROLLOUT + start)
    end = start + 31 + torch.randint(0, PSI_ROLLOUT - 31, (n_reset,), generator=g)
    end = torch.minimum(end, end_i - 1)                  # the task asserts end < end_i
    end = torch.maximum(end, start)
    state = torch.randn(n_reset, PSI_ROLLOUT + 8, REFS_W, generator=g)
    return data_id, start, end, end_i, state, mel


def test_psi_buffer_update_identical():
    """Run the real PSI update against both stores; buffers must come out equal
    at every valid cell, ref_reward (shared shape) identical, and the update must
    have WRITTEN something or the comparison is vacuous."""
    pad_refs, rag_refs = _psi_stores()
    data_id, start, end, end_i, state, mel = _psi_batch(seed=9)
    rr_pad = torch.zeros(len(PSI_LENGTHS), TOPK, max(PSI_LENGTHS))
    rr_pad[:, 0, :] = 1.0
    rr_rag = rr_pad.clone()

    w_pad = psi_buffer_update(data_id, start, end, end_i, state, rr_pad, pad_refs,
                              mel, PSI_ROLLOUT)
    w_rag = psi_buffer_update(data_id, start, end, end_i, state, rr_rag, rag_refs,
                              mel, PSI_ROLLOUT)
    assert w_pad == w_rag
    assert w_pad > 0, "vacuous: the batch produced no PSI writes"
    torch.testing.assert_close(rr_pad, rr_rag, rtol=0, atol=0)
    for m, n in enumerate(PSI_LENGTHS):
        t = torch.arange(n)
        mm = torch.full((n,), m)
        for s in range(TOPK):
            ss = torch.full((n,), s)
            torch.testing.assert_close(rag_refs[(mm, ss, t)], pad_refs[mm, ss, t],
                                       rtol=0, atol=0)
    # and the written slots are no longer the mocap copy: slot 1 differs from slot 0
    # somewhere, on both stores alike
    changed = int((rr_pad[:, 1:, :] > 0).sum())
    assert changed == w_pad


def test_psi_update_repeated_batches_stay_identical():
    """Several batches in a row, as in training: the weakest-slot choice depends
    on the ref_reward left by the previous batch, so divergence would compound."""
    pad_refs, rag_refs = _psi_stores()
    rr_pad = torch.zeros(len(PSI_LENGTHS), TOPK, max(PSI_LENGTHS)); rr_pad[:, 0, :] = 1.0
    rr_rag = rr_pad.clone()
    total = 0
    for seed in range(20, 26):
        data_id, start, end, end_i, state, mel = _psi_batch(seed=seed)
        w_pad = psi_buffer_update(data_id, start, end, end_i, state, rr_pad, pad_refs, mel, PSI_ROLLOUT)
        w_rag = psi_buffer_update(data_id, start, end, end_i, state, rr_rag, rag_refs, mel, PSI_ROLLOUT)
        assert w_pad == w_rag
        total += w_pad
        rr_pad[:, 1:, :] *= (1 - 1e-5); rr_rag[:, 1:, :] *= (1 - 1e-5)   # the task's decay
    assert total > 0
    torch.testing.assert_close(rr_pad, rr_rag, rtol=0, atol=0)
    for m, n in enumerate(PSI_LENGTHS):
        t = torch.arange(n); mm = torch.full((n,), m)
        for s in range(TOPK):
            torch.testing.assert_close(rag_refs[(mm, torch.full((n,), s), t)],
                                       pad_refs[mm, s, t], rtol=0, atol=0)


# ---- fact 3: reads past a clip fail, they do not read the next clip ---------
def test_out_of_range_frame_raises(stores):
    L = stores["lengths"]
    with pytest.raises(IndexError):
        stores["rag_data"][(torch.tensor([0]), torch.tensor([L[0]]))]      # == length
    with pytest.raises(IndexError):
        stores["rag_data"][(torch.tensor([3]), torch.tensor([-1]))]
    with pytest.raises(IndexError):
        stores["rag_refs"][(torch.tensor([1]), torch.tensor([0]), torch.tensor([1]))]


def test_out_of_range_slot_raises(stores):
    with pytest.raises(IndexError):
        stores["rag_refs"][(torch.tensor([0]), torch.tensor([TOPK]), torch.tensor([0]))]


def test_bounds_check_default_cpu_on_gpu_off():
    r = RaggedMotion([3, 4])
    assert r.check_bounds is True
    if torch.cuda.is_available():
        assert RaggedMotion([3, 4], device="cuda").check_bounds is False
        os.environ["RAGGED_CHECK_BOUNDS"] = "1"
        try:
            assert RaggedMotion([3, 4], device="cuda").check_bounds is True
        finally:
            del os.environ["RAGGED_CHECK_BOUNDS"]


# ---- construction guards ----------------------------------------------------
def test_write_clip_guards():
    r = RaggedMotion([3, 4])
    with pytest.raises(ValueError):
        r.write_clip(0, torch.zeros(2, 5))          # wrong length
    r.write_clip(0, torch.zeros(3, 5))
    with pytest.raises(ValueError):
        r.write_clip(1, torch.zeros(4, 6))          # wrong width
    with pytest.raises(ValueError):
        r.write_clip(0, torch.zeros(3, 5))          # twice
    with pytest.raises(RuntimeError):
        r.assert_complete()                         # clip 1 missing
    r.write_clip(1, torch.zeros(4, 5))
    assert r.assert_complete() is r


def test_logical_shape_and_bytes(stores):
    """intermimic.py allocates ref_reward from hoi_refs.shape[0:3] and prints
    element_size()*nelement(): the first must be the PADDED shape, the second the
    bytes actually held."""
    rag, pad = stores["rag_refs"], stores["pad_refs"]
    assert tuple(rag.shape) == tuple(pad.shape)
    assert tuple(stores["rag_data"].shape) == tuple(stores["pad_data"].shape)
    assert rag.nelement() == sum(stores["lengths"]) * TOPK * REFS_W
    assert rag.padded_nelement() == pad.nelement()
    assert rag.is_cuda is False and rag.dtype == pad.dtype


def test_scan_clip_lengths(tmp_path):
    lens = [5, 2, 9]
    paths = []
    for i, n in enumerate(lens):
        p = tmp_path / f"sub2_obj_{i:03d}.pt"
        torch.save(torch.zeros(n, 591), p)
        paths.append(str(p))
    assert scan_clip_lengths(paths) == lens
    assert scan_clip_lengths(paths, startk=1) == [4, 1, 8]
    assert scan_clip_lengths(paths, startk=1, initk=15) == [19, 16, 23]
    with pytest.raises(ValueError):
        scan_clip_lengths(paths, startk=2)          # the 2-frame clip goes empty


def test_write_clip_path_guard():
    """Two equal-length clips swapped pass the length check; the path check is
    what catches a loop that no longer walks the scanned list in order."""
    paths = ["/x/a.pt", "/x/b.pt"]
    r = RaggedMotion([3, 3], paths=paths)
    with pytest.raises(ValueError):
        r.write_clip(0, torch.zeros(3, 2))                     # path required
    with pytest.raises(ValueError):
        r.write_clip(0, torch.zeros(3, 2), path="/x/b.pt")     # swapped
    r.write_clip(0, torch.zeros(3, 2), path="/x/a.pt")
    r.write_clip(1, torch.zeros(3, 2), path="/x/b.pt")
    assert r.assert_complete() is r
    with pytest.raises(ValueError):
        RaggedMotion([3, 3], paths=["/x/a.pt"])               # count mismatch


# ---- the loader, end to end on REAL files ----------------------------------
REAL_DIR = os.path.join(REPO, "InterAct/OMOMO_new")


def _loader_frame_axis(path, startk):
    """Transcription of every frame-axis operation in intermimic.py _load_motion
    (initk=0): drop startk frames, then every component is a column slice of
    that, and every velocity is a finite difference with ONE zero row prepended.
    So the output has exactly file_T - startk rows. Column content is irrelevant
    to storage; a few representative columns stand in for the 1211."""
    hoi = torch.load(path)[startk:]
    root_pos = hoi[:, 0:3].clone()
    vel = (root_pos[1:] - root_pos[:-1]) * 30.
    vel = torch.cat((torch.zeros((1, vel.shape[-1])), vel), dim=0)
    dof = hoi[:, 9:9 + 153].clone()
    return torch.cat((root_pos, vel, dof), dim=-1)


@pytest.mark.skipif(not os.path.isdir(REAL_DIR), reason="bundled OMOMO clips not present")
@pytest.mark.parametrize("startk", [0, 1])
def test_loader_two_pass_on_real_clips(startk):
    """scan_clip_lengths sizes the store from the files; the processing pass
    (transcribed) must fill every slot with exactly that many rows, and the
    result must equal the padded assembly of the same clips."""
    paths = sorted(os.path.join(REAL_DIR, f) for f in os.listdir(REAL_DIR) if f.endswith(".pt"))[:8]
    assert len(paths) >= 3
    lengths = scan_clip_lengths(paths, startk=startk, initk=0)
    for p, n in zip(paths, lengths):
        assert n == int(torch.load(p).shape[0]) - startk
    r_data = RaggedMotion(lengths, paths=paths)
    r_refs = RaggedMotion(lengths, topk=TOPK, paths=paths)
    clips = []
    for idx, data_path in enumerate(paths):             # same list, same order
        c = _loader_frame_axis(data_path, startk)
        r_data.write_clip(idx, c, path=data_path)
        r_refs.write_clip(idx, c[:, :7], path=data_path)
        clips.append(c)
    r_data.assert_complete(); r_refs.assert_complete()
    pad = padded_like_loader(clips)
    pad_refs = padded_like_loader([c[:, :7] for c in clips], topk=TOPK)
    assert tuple(r_data.shape) == tuple(pad.shape)
    for m, n in enumerate(lengths):
        t = torch.arange(n); mm = torch.full((n,), m)
        torch.testing.assert_close(r_data[(mm, t)], pad[mm, t], rtol=0, atol=0)
        torch.testing.assert_close(r_refs[(mm, torch.full((n,), 2), t)],
                                   pad_refs[mm, 2, t], rtol=0, atol=0)
    assert r_data.nelement() < pad.nelement()          # it actually saved something


# ---- source guard: the hunk in intermimic.py keeps the contract ------------
TASK_SRC = os.path.join(REPO, "isaacgym/src/intermimic/env/tasks/intermimic.py")


def test_loader_source_contract():
    """The two passes must walk `motion_file` positionally, write inside that
    loop by (idx, data_path), seal the store after, and refuse initk != 0.
    Read the source rather than importing it: intermimic.py needs isaacgym."""
    import re
    src = open(TASK_SRC).read()
    body = src[src.index("def _load_motion("):src.index("def create_component_stat(")]
    assert "scan_clip_lengths(motion_file, startk=startk, initk=initk)" in body
    assert re.search(r"^\s*for idx, data_path in enumerate\(motion_file\):", body, re.M)
    assert body.count("write_clip(idx, ") == 2
    assert "path=data_path" in body
    assert body.index("for idx, data_path in enumerate(motion_file)") < body.index("write_clip(idx, ")
    assert body.index("write_clip(idx, ") < body.index("assert_complete()")
    assert "if initk != 0:" in body and "NotImplementedError" in body
    # the flag is read as an attribute, never defaulted: a missing attribute raises
    assert "getattr(self, '_ragged_motion'" not in body
    assert "ragged = self._ragged_motion" in body
    # and the cfg key is validated, so a misspelling cannot silently mean padded
    assert "'raggedMotionData'" in src[src.index("KNOWN_ENV_KEYS"):src.index("def _validate_env_config")]
    # the list is wrapped BEFORE the scan, so scan and loop see the same object
    assert body.index("motion_file = [motion_file]") < body.index("scan_clip_lengths(motion_file")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_gpu_store_matches(lengths):
    clips = make_clips(lengths, DATA_W, seed=11)
    pad = padded_like_loader(clips).cuda()
    rag = ragged_from(clips, device="cuda")
    m, t = valid_index(lengths, 300, seed=12)
    torch.testing.assert_close(rag[(m.cuda(), t.cuda())], pad[m.cuda(), t.cuda()],
                               rtol=0, atol=0)
