# Source-Teacher Experiments

One-page reference for the source-specific teacher work and its follow-on
experiments. Each experiment lives on its own branch so runs don't collide.

**Source-teacher idea:** train one body-conditioned policy per motion SOURCE
(sub2's motion driving many bodies), then distill the per-source teachers into a
single student — an alternative to the staged multi-source curriculum.

- **source** `S` = `dataSub` — whose mocap MOTION is the reference.
- **body** `B` = `subjectBodies` — whose morphology the policy controls (betas-conditioned).
- **held-out bodies** = {sub4, sub10, sub13, sub16} — never TARGET bodies; still valid motion sources.
- **synthetic bodies** = sub100–139 (27 "inhull" within the real shape range + 12 "extrapolated" beyond it). sub121 dropped (near-duplicate of held-out sub13).

All slurm scripts run from the repo root; conda env `intermimic-gym2`.

---

## Branch map

| Branch | What's on it |
|---|---|
| `source-teacher` | The 17 baseline teachers: `omomo_teacher_src{1..17}_xf_aug` (transformer + neutral betas + synthetic bodies, no staging) |
| `source-teacher-drop-sub121` | sub121 removed from all 17 (39 synth) + generator held-out fix + `analyze_synthetic_bodies.py` |
| `source-teacher-staged` | Staged + adaptive-LR arms for src2/src6 + reward diagnostics ON by default |
| `source-teacher-distill` | Distillation wiring (transformer-teacher builder patch) + 2 source-set variants |

Each branch descends from the previous, so `-distill` and `-staged` both include the sub121 fix.

---

## 1. Baseline teachers (17) — `source-teacher-drop-sub121`

Run the corrected (sub121-free) set:
```bash
for s in $(seq 1 17); do sbatch slurm_teacher_src${s}_xf_aug.sh; done
```
Saves to `checkpoints/smplx_teacher_src{S}_xf_aug/nn/`.

No-aug ablations also exist: `slurm_teacher_src{S}.sh` (MLP), `slurm_teacher_src{S}_xf.sh` (transformer, no synthetic).

---

## 2. Staged + adaptive-LR — `source-teacher-staged`

Both attack the near-linear reward curve (cause: **multiplicative AND-gate**
`reward = rb·ro·rig·rcg`, mean over 52 bodies, constant LR 2e-5).

**Staged** — fixed source, 13 real + 27 inhull live from stage 0, then fold the
12 extrapolated synthetic bodies in one at a time (13 stages, resuming each):
```bash
sbatch slurm_teacher_src2_staged.sh
sbatch slurm_teacher_src6_staged.sh
```
Driver: `scripts/staged_source_teacher_runner.py` (weight-mask + `maskDeadEnvs`;
`--resume` skips finished stages). Final policy: `smplx_teacher_src{S}_staged_s12`.

**Adaptive-LR** — same env, `lr_schedule: adaptive` (KL 0.008, start 2e-4):
```bash
sbatch slurm_teacher_src2_xf_aug_adlr.sh
sbatch slurm_teacher_src6_xf_aug_adlr.sh
```

**Reward diagnostics are ON by default on this branch** (`REWARD_BREAKDOWN=0`
to silence). Watch the `by body:` / `by object:` lines to see which reward
factor pins the product.

---

## 3. Distillation — `source-teacher-distill`

One transformer student imitates the per-source teachers (`InterMimic_All`
selects a teacher per env by source subid). Two source-set variants:

```bash
sbatch slurm_distill_source_noheldout.sh   # 13 non-held-out sources
sbatch slurm_distill_source_no14.sh        # all 17 except sub14
```
Each collects teacher checkpoints (`collect_source_teachers.py`) then runs
`run_distill.py --task InterMimic_All`. Students → `checkpoints/smplx_student_source_xf_aug_{variant}/`.

**Requires the teachers to be trained first.** This is the FIRST distill from
transformer teachers — the first launch smoke-tests the vmap teacher-query path
and obs sizes (both error loudly at startup if wrong).

---

## Evaluation

```bash
sh scripts/eval_one.sh <run>            # e.g. src9_xf_aug ; latest ckpt, held-out + synth
DRY=1 sh scripts/eval_one.sh <run>      # preview resolution, don't submit
```
Metrics valid only in `Start` state-init. Held-out test set = {sub10, sub16, sub13}
(sub4 excluded — MJCF sim-crasher). Reports land in `eval_results/`.

---

## Status

Everything in experiments 2–3 is **built and validated locally (parse / dry-run /
compile) but not yet cluster-run** — no Isaac Gym off-cluster. Before more reward
surgery, ground the flat curve in behavior: read the per-body breakdown + watch a
rollout to confirm the bottleneck.

## Known body issues (held-out eval)
- **sub4** — only sim-crasher; bad MJCF; excluded from eval (no retrain fix).
- **sub13** — contaminated by sub121 (fixed in training going forward); its held-out number was unreliable.
- **sub16** — genuinely hard, root cause UNKNOWN (beta-distance refuted; bodies verified correct → failure is downstream in policy/conditioning). Under investigation.
