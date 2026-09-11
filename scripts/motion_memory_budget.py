#!/usr/bin/env python3
"""How much host RAM will a teacher's reference motion take -- padded vs ragged?

The task loads every clip of every source in `dataSub`, once per target body in
`subjectBodies` (per-body contact retargeting), and holds two tensors for the
whole run:

    hoi_data  (num_motions, T, 1211)        the reference the policy tracks
    hoi_refs  (num_motions, psi, T, 332)    the PSI init buffer, psi slots

With PADDED storage T is the LONGEST clip in the set, so a 150-frame clip costs
as much as a 652-frame one. With RAGGED storage (env cfg `raggedMotionData:
true`) each clip costs exactly its own length. This script measures both from
the clip files themselves, so an arm's `#SBATCH --mem` comes from a number and
not a guess. Nothing here touches the simulator; it only reads tensor shapes.

    python3 scripts/motion_memory_budget.py --motion-dir ~/new_one/OMOMO_new \\
        --sources sub1 sub2 sub3 sub5 sub6 sub7 sub8 sub9 sub11 sub12 sub14 sub15 sub17 \\
        --bodies 43 --psi 3

    # or read dataSub / subjectBodies / physicalBufferSize straight from an arm:
    python3 scripts/motion_memory_budget.py --motion-dir ~/new_one/OMOMO_new \\
        --cfg isaacgym/src/intermimic/data/cfg/omomo_teacher_g3_omomo_geoall_srcall13__f0.yaml

The "total RAM" column applies the model validated on 2026-09-09 against three
measured MaxRSS peaks (residuals under 5 GiB):

    total RAM = 14.2 GiB + 2.02 * motion GiB          (padded, as measured)

The 2.02 is mostly the load-time transient: the padded loader keeps a Python
list of every clip AND the stacked copy alive at once. The ragged loader
preallocates once and writes each clip in place, so its transient is one clip,
and its steady state is 14.2 + 1.0 * motion. That 1.0 has NOT been measured on
a real run yet -- until it is, provision the first ragged arm with the 2.02
factor and read `[mem] motion tensors:` and MaxRSS from the job.
"""
import argparse
import os
import sys

import torch

# Column widths of the two run-time tensors (intermimic.py: ref_hoi_obs_size and
# the hoi_ref concatenation in _load_motion). Fixed by the SMPL-X rig.
HOI_DATA_WIDTH = 7 + 51 * 6 + 52 * 13 + 13 + 52 * 3 + 52 + 1   # 1211
HOI_REFS_WIDTH = 3 + 4 + 3 + 3 + 153 + 153 + 3 + 4 + 3 + 3       # 332
BYTES = 4                                                          # float32
GIB = 1024 ** 3

# Validated 2026-09-09 (see docstring). Base is everything that is not motion:
# the sim, the policy, PhysX host mirrors, Python.
RAM_BASE_GIB = 14.2
RAM_PADDED_FACTOR = 2.02


def subject_of(fname):
    """'sub12_largetable_003.pt' -> 'sub12'."""
    return fname.split("_", 1)[0]


def clip_lengths(motion_dir, sources):
    """{source: [T per clip]} by opening every clip and reading its frame count.

    Reads the tensor header only in effect (torch.load then .shape), so it is
    I/O-bound; 4.4k clips take ~10 s from a warm cache.
    """
    want = set(sources)
    out = {s: [] for s in sources}
    files = sorted(f for f in os.listdir(motion_dir) if f.endswith(".pt"))
    for f in files:
        s = subject_of(f)
        if s not in want:
            continue
        t = torch.load(os.path.join(motion_dir, f), map_location="cpu", weights_only=True)
        out[s].append(int(t.shape[0]))
    missing = [s for s in sources if not out[s]]
    if missing:
        raise SystemExit(f"ERROR: no clips for {missing} under {motion_dir} -- "
                         f"wrong dir or misspelled subject")
    return out


def budget(lengths_by_source, n_bodies, psi):
    """Padded vs ragged bytes for the whole (bodies x clips) set."""
    lengths = [t for s in lengths_by_source.values() for t in s]
    n_clips = len(lengths)
    max_t = max(lengths)
    sum_t = sum(lengths)
    row_bytes = (HOI_DATA_WIDTH + psi * HOI_REFS_WIDTH) * BYTES
    padded = n_bodies * n_clips * max_t * row_bytes
    ragged = n_bodies * sum_t * row_bytes
    return dict(n_clips=n_clips, max_t=max_t, sum_t=sum_t,
                mean_t=sum_t / n_clips, padded=padded, ragged=ragged)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--motion-dir", required=True,
                   help="source clip dir (the arm's motion_file), e.g. InterAct/OMOMO_new")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--sources", nargs="+", help="dataSub entries, e.g. sub1 sub12")
    g.add_argument("--cfg", help="env yaml to read dataSub/subjectBodies/physicalBufferSize from")
    p.add_argument("--bodies", type=int, help="number of target bodies (len(subjectBodies))")
    p.add_argument("--psi", type=int, help="physicalBufferSize (PSI slots)")
    a = p.parse_args(argv)

    if a.cfg:
        import yaml
        env = yaml.safe_load(open(a.cfg))["env"]
        sources = list(env["dataSub"])
        n_bodies = len(env["subjectBodies"]) if a.bodies is None else a.bodies
        psi = int(env.get("physicalBufferSize", 1)) if a.psi is None else a.psi
        print(f"from {os.path.basename(a.cfg)}: {len(sources)} sources, "
              f"{n_bodies} bodies, psi {psi}")
    else:
        sources = a.sources
        if a.bodies is None or a.psi is None:
            p.error("--bodies and --psi are required with --sources")
        n_bodies, psi = a.bodies, a.psi

    lengths = clip_lengths(os.path.expanduser(a.motion_dir), sources)

    print(f"\n{'source':>7} {'clips':>6} {'sum T':>8} {'max T':>6} {'mean T':>7}")
    for s in sources:
        ls = lengths[s]
        print(f"{s:>7} {len(ls):>6} {sum(ls):>8} {max(ls):>6} {sum(ls) / len(ls):>7.1f}")

    b = budget(lengths, n_bodies, psi)
    print(f"\n{len(sources)} sources x {n_bodies} bodies = "
          f"{b['n_clips'] * n_bodies:,} motions; {b['n_clips']} clips, "
          f"max T {b['max_t']}, mean T {b['mean_t']:.1f}, sum T {b['sum_t']:,}")
    print(f"row = {HOI_DATA_WIDTH} + {psi} x {HOI_REFS_WIDTH} floats = "
          f"{(HOI_DATA_WIDTH + psi * HOI_REFS_WIDTH) * BYTES:,} bytes/frame")
    pad_gib, rag_gib = b["padded"] / GIB, b["ragged"] / GIB
    print(f"\n{'storage':>8} {'motion GiB':>11} {'RAM GiB (2.02x model)':>22} {'RAM GiB (1.0x)':>15}")
    print(f"{'padded':>8} {pad_gib:>11.1f} {RAM_BASE_GIB + RAM_PADDED_FACTOR * pad_gib:>22.0f} "
          f"{RAM_BASE_GIB + pad_gib:>15.0f}")
    print(f"{'ragged':>8} {rag_gib:>11.1f} {RAM_BASE_GIB + RAM_PADDED_FACTOR * rag_gib:>22.0f} "
          f"{RAM_BASE_GIB + rag_gib:>15.0f}")
    print(f"\npadding waste: {pad_gib / rag_gib:.2f}x  "
          f"(padded / ragged; = max T / mean T)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
