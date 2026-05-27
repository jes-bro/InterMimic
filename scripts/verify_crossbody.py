#!/usr/bin/env python3
"""Verify cross-body assignment happens at env reset.

Loads a fresh InterMimic env exactly the way the training script does,
runs one env reset, then dumps for each env:
  (env_idx, env_body_subject, motion_source_subject, motion_target_subject)

If env_body_subject == motion_source_subject for all envs, we are NOT
doing cross-body retargeting (the env code is somehow coupling them).
If they differ for a meaningful fraction, cross-body retargeting is
real.

Prints first 30 envs' assignments + aggregate histogram. No checkpoint
needed (env init is enough; no policy involved).

Run on the cluster:
    python scripts/verify_crossbody.py
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

# Match what intermimic.run does
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "isaacgym/src"))
sys.path.insert(0, str(REPO_ROOT))

from intermimic.utils.config import load_cfg, get_args, parse_sim_params
from intermimic.utils.parse_task import parse_task


def main():
    # Pretend we're being called like:
    # python -u -m intermimic.run --task InterMimic --cfg_env <multibody yaml> \
    #     --cfg_train <multibody train yaml> --num_envs 64
    sys.argv = [
        sys.argv[0],
        "--task", "InterMimic",
        "--cfg_env", "isaacgym/src/intermimic/data/cfg/omomo_train_multibody.yaml",
        "--cfg_train", "isaacgym/src/intermimic/data/cfg/train/rlg/omomo_multibody.yaml",
        "--headless",
        "--num_envs", "64",
    ]

    args = get_args()
    cfg, cfg_train, _ = load_cfg(args)
    sim_params = parse_sim_params(args, cfg, cfg_train)
    task, env = parse_task(args, cfg, cfg_train, sim_params)

    # Force a reset of all envs so data_id, _env_subject_idx are populated
    # consistently. parse_task already does an initial reset.
    env_ids = task._env_subject_idx.new_tensor(list(range(task.num_envs)))

    # subjectBodies is a list of strings like ['sub10', 'sub17', ...]
    # _env_subject_idx[i] indexes into this list
    subject_bodies = task.subject_bodies  # list of strings
    body_subj_per_env = [
        int(subject_bodies[task._env_subject_idx[i].item()][3:])
        for i in range(task.num_envs)
    ]

    # data_id[i] indexes into the motion pool
    source_per_env = task.source_subject_index[task.data_id].tolist()
    target_per_env = task.target_subject_index[task.data_id].tolist()
    # object_id[motion_id] tells us which object slot a motion uses
    object_per_env = [task.object_name[task.object_id[mid.item()].item()]
                      for mid in task.data_id]

    print("\n=== First 30 envs ===")
    print(f"{'env':>4} {'body':>6} {'src':>8} {'tgt':>8} {'object':>15}  {'cross?'}")
    print("-" * 60)
    for i in range(min(30, task.num_envs)):
        cross = "yes" if body_subj_per_env[i] != source_per_env[i] else "no"
        print(f"{i:>4} sub{body_subj_per_env[i]:<3} sub{source_per_env[i]:<5} sub{target_per_env[i]:<5} {object_per_env[i]:>15}  {cross}")

    # Aggregate
    n = task.num_envs
    n_cross = sum(1 for i in range(n) if body_subj_per_env[i] != source_per_env[i])
    print(f"\n=== Aggregate over {n} envs ===")
    print(f"cross-body (body != motion_source): {n_cross}/{n} = {n_cross/n*100:.1f}%")
    print(f"identity:                            {n-n_cross}/{n} = {(n-n_cross)/n*100:.1f}%")

    # Per-object breakdown — distinguishes multi-subject objects (where both
    # body and source vary) from single-subject objects (where only body varies).
    print("\n=== Per-object breakdown ===")
    print(f"{'object':>15} {'envs':>6} {'#bodies':>9} {'#sources':>10} {'%cross':>8}")
    print("-" * 60)
    per_obj_bodies = defaultdict(set)
    per_obj_sources = defaultdict(set)
    per_obj_count = Counter()
    per_obj_cross = Counter()
    for i in range(n):
        obj = object_per_env[i]
        per_obj_bodies[obj].add(body_subj_per_env[i])
        per_obj_sources[obj].add(source_per_env[i])
        per_obj_count[obj] += 1
        if body_subj_per_env[i] != source_per_env[i]:
            per_obj_cross[obj] += 1
    for obj in sorted(per_obj_count.keys()):
        nb = len(per_obj_bodies[obj])
        ns = len(per_obj_sources[obj])
        c = per_obj_count[obj]
        x = per_obj_cross[obj]
        print(f"{obj:>15} {c:>6} {nb:>9} {ns:>10} {x/c*100:>7.1f}%")

    # Detailed pair counts
    pair_counts = Counter(zip(body_subj_per_env, source_per_env))
    print("\n=== (body, motion_source) pair counts ===")
    for (b, s), c in sorted(pair_counts.items()):
        marker = "  (identity)" if b == s else ""
        print(f"  body=sub{b:<2} motion_source=sub{s:<3}  count={c}{marker}")


if __name__ == "__main__":
    main()
