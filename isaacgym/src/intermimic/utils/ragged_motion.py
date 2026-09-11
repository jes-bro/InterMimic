#!/usr/bin/env python3
"""Ragged (unpadded) storage for the reference-motion tensors.

The task keeps two big tensors for the life of a run:

    hoi_data  (num_motions, T, 1211)        reference frames the policy tracks
    hoi_refs  (num_motions, psi, T, 332)    PSI init buffer, `psi` slots per frame

Both were PADDED: T is the longest clip in the set, so every clip pays for the
longest one. With per-body retargeting (43 bodies x 3356 clips for the 13-source
OMOMO teacher) and a 652-frame outlier among 191-frame-mean clips, that padding
is 3.4x the data -- 774 GiB where the frames themselves are 226 GiB
(scripts/motion_memory_budget.py). A whole simurgh node is 1.5 TB.

This class stores clips back-to-back in one flat buffer and answers the SAME
index tuples the padded tensors did, so no reader changes:

    hoi_data[(m, t)]             -> (B, 1211)      rows offsets[m] + t
    hoi_data[(m, t, slice)]      -> (B, width)
    hoi_refs[(m, s, t)]          -> (B, 332)       slot s of frame t
    hoi_refs[(m, s, t, slice)]   -> (B, width)
    hoi_refs[(m, s, t)] = value                    (the PSI write)

`m`, `s`, `t` are long tensors of one common shape (or Python ints). A frame
index at or past its clip's length is an ERROR here, never a silent read of the
next clip's data; the padded tensor would have returned zeros for those, and
every reader in intermimic.py clamps or resets before that can happen
(_compute_observations_iter clamps to mel-1, compute_humanoid_reset resets at
progress_buf >= mel-1, psi_update writes only where mel - j >= rollout_length,
play_dataset_step clamps). The check is free when the buffer lives on CPU
(cpuMotionData), where the indices are already host-side; on a GPU buffer it
costs a device sync per gather, so it is off there unless RAGGED_CHECK_BOUNDS=1.

Allocation is ONE buffer up front (lengths come from a shape-only pass over the
files, `scan_clip_lengths`), and each clip is written in place as it is
processed. The padded loader held a Python list of every processed clip AND the
stacked copy at its peak; that transient is what made resident memory ~2x the
motion size. Here the transient is one clip.

Differential tests: tests/test_ragged_motion.py pins every gather, the setitem,
and a full psi_buffer_update against the padded tensor built from the same clips.
"""
import os

import torch


def scan_clip_lengths(paths, startk=0, initk=0):
    """Frame count each clip will have AFTER the loader's startk drop and initk
    prepend, read by opening every file for its shape. Pure I/O; the second pass
    reads the same files again, from the page cache."""
    lengths = []
    for p in paths:
        t = torch.load(p, map_location="cpu")
        n = int(t.shape[0]) - startk + initk
        if n <= 0:
            raise ValueError(f"[ragged] {p}: {int(t.shape[0])} frames, startk={startk} "
                             f"leaves {n} -- empty clip")
        lengths.append(n)
    return lengths


class RaggedMotion:
    """See module docstring. Construct with per-clip lengths; the buffer is
    allocated on the first write_clip (that is where width and dtype are known),
    and assert_complete() refuses to hand out a half-written store."""

    def __init__(self, lengths, topk=None, device="cpu", check_bounds=None, paths=None):
        if not lengths:
            raise ValueError("[ragged] no clips")
        # Optional: the file each slot was sized from. write_clip(i, clip, path=...)
        # then checks the loop is filling the slot the scan measured -- the guard
        # against a reordered / filtered loop assigning lengths to the wrong clips
        # (two equal-length clips swapped would pass the length check alone).
        self.paths = None if paths is None else list(paths)
        if self.paths is not None and len(self.paths) != len(lengths):
            raise ValueError(f"[ragged] {len(self.paths)} paths for {len(lengths)} lengths")
        self.lengths = torch.as_tensor(list(lengths), dtype=torch.long)     # CPU
        if bool((self.lengths <= 0).any()):
            raise ValueError("[ragged] every clip length must be positive")
        # Exclusive prefix sum: clip i occupies rows [offsets[i], offsets[i] + lengths[i]).
        self.offsets = torch.cumsum(self.lengths, 0) - self.lengths
        self.num_motions = int(self.lengths.shape[0])
        self.max_length = int(self.lengths.max())
        self.total_frames = int(self.lengths.sum())
        self.topk = None if topk is None else int(topk)
        if self.topk is not None and self.topk < 1:
            raise ValueError(f"[ragged] topk must be >= 1, got {topk}")
        self.device = torch.device(device)
        self._offsets_dev = self.offsets.to(self.device)
        self._lengths_dev = self.lengths.to(self.device)
        if check_bounds is None:
            check_bounds = (self.device.type == "cpu"
                            or os.environ.get("RAGGED_CHECK_BOUNDS", "") == "1")
        self.check_bounds = bool(check_bounds)
        self.data = None            # allocated on first write_clip
        self._written = torch.zeros(self.num_motions, dtype=torch.bool)

    # ---- construction -------------------------------------------------------
    def write_clip(self, i, clip, path=None):
        """Store clip i, shape (lengths[i], width). For a topk store the clip is
        copied into EVERY slot, matching the padded loader's `.repeat(1, topk, 1, 1)`
        (slot 0 is the mocap reference; PSI later overwrites slots 1: in place).
        `path`, when the store was built with paths, must be the file slot i was
        sized from."""
        i = int(i)
        if self.paths is not None:
            if path is None:
                raise ValueError("[ragged] this store was built with paths; pass path=")
            if path != self.paths[i]:
                raise ValueError(f"[ragged] slot {i} was sized from {self.paths[i]} but "
                                 f"is being written from {path} -- loop order differs "
                                 f"from the length scan")
        n = int(self.lengths[i])
        if clip.dim() != 2:
            raise ValueError(f"[ragged] clip {i}: expected (T, width), got {tuple(clip.shape)}")
        if int(clip.shape[0]) != n:
            raise ValueError(f"[ragged] clip {i}: {int(clip.shape[0])} frames but the "
                             f"length scan said {n} -- file changed between passes?")
        if self.data is None:
            width = int(clip.shape[1])
            shape = ((self.total_frames, width) if self.topk is None
                     else (self.total_frames, self.topk, width))
            # empty, not zeros: every row is written before assert_complete() lets
            # the store be used, and zeroing 200+ GiB just to overwrite it is slow.
            self.data = torch.empty(shape, dtype=clip.dtype, device=self.device)
            self.width = width
        elif int(clip.shape[1]) != self.width:
            raise ValueError(f"[ragged] clip {i}: width {int(clip.shape[1])} != {self.width}")
        if bool(self._written[i]):
            raise ValueError(f"[ragged] clip {i} written twice")
        off = int(self.offsets[i])
        src = clip.to(self.device, dtype=self.data.dtype)
        if self.topk is None:
            self.data[off:off + n] = src
        else:
            self.data[off:off + n] = src.unsqueeze(1)      # broadcast over slots
        self._written[i] = True

    def assert_complete(self):
        missing = int((~self._written).sum())
        if self.data is None or missing:
            raise RuntimeError(f"[ragged] {missing}/{self.num_motions} clips never written")
        return self

    # ---- tensor-like surface the task relies on ----------------------------
    @property
    def shape(self):
        """The LOGICAL padded shape, so `hoi_refs.shape[0..2]` (ref_reward's
        allocation) and the [mem] print keep working unchanged."""
        if self.topk is None:
            return (self.num_motions, self.max_length, self.width)
        return (self.num_motions, self.topk, self.max_length, self.width)

    @property
    def is_cuda(self):
        return self.device.type == "cuda"

    @property
    def dtype(self):
        return self.data.dtype

    def element_size(self):
        return self.data.element_size()

    def nelement(self):
        """Elements actually STORED (not the logical padded count)."""
        return self.data.nelement()

    def padded_nelement(self):
        n = self.num_motions * self.max_length * self.width
        return n if self.topk is None else n * self.topk

    # ---- indexing -----------------------------------------------------------
    def _parse(self, idx):
        if not isinstance(idx, tuple):
            raise TypeError("[ragged] index with a tuple: (m, t[, cols]) or (m, s, t[, cols])")
        n_int = 3 if self.topk is not None else 2
        cols = slice(None)
        if len(idx) == n_int + 1:
            cols = idx[-1]
            if not isinstance(cols, slice):
                raise TypeError(f"[ragged] trailing index must be a slice, got {type(cols)}")
            idx = idx[:-1]
        if len(idx) != n_int:
            raise TypeError(f"[ragged] expected {n_int} index tensors, got {len(idx)}")
        parts = [torch.as_tensor(x, dtype=torch.long, device=self.device) for x in idx]
        if self.topk is None:
            m, t = parts
            s = None
        else:
            m, s, t = parts
        rows = self._offsets_dev[m] + t
        if self.check_bounds:
            bad = (t < 0) | (t >= self._lengths_dev[m])
            if bool(bad.any()):
                b = bad.nonzero()[0]
                mm, tt = int(m[tuple(b)]), int(t[tuple(b)])
                raise IndexError(f"[ragged] frame {tt} of motion {mm} (length "
                                 f"{int(self.lengths[mm])}) -- a reader indexed past "
                                 f"the clip; the padded tensor would have returned zeros")
            if s is not None:
                bad_s = (s < 0) | (s >= self.topk)
                if bool(bad_s.any()):
                    raise IndexError(f"[ragged] PSI slot out of range (topk={self.topk})")
        return rows, s, cols

    def __getitem__(self, idx):
        rows, s, cols = self._parse(idx)
        if s is None:
            return self.data[rows, cols]
        return self.data[rows, s, cols]

    def __setitem__(self, idx, value):
        rows, s, cols = self._parse(idx)
        if s is None:
            self.data[rows, cols] = value
        else:
            self.data[rows, s, cols] = value

    def __repr__(self):
        return (f"RaggedMotion(motions={self.num_motions}, frames={self.total_frames}, "
                f"max_length={self.max_length}, topk={self.topk}, device={self.device})")
