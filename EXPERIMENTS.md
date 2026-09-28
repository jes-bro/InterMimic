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
- **sub13** — contaminated by sub121 (fixed in training going forward); its held-out number was unreliable. *(2026-08-09: still leaks via sub123/sub125 — see the log below.)*
- **sub16** — genuinely hard, root cause UNKNOWN (beta-distance refuted; bodies verified correct → failure is downstream in policy/conditioning). Under investigation. *(2026-08-09: two more kinematic hypotheses refuted — see the log below.)*

---

# Log: 2026-08-07 → 08-09 — evals, post-mortems, and the gen-2 grid

Everything below is on `physx-buffer-headroom`, merged to `main` at `a4d64bc`.
Eval CSVs + figures live in `~/Downloads/latestresultsmorefinishedaug{8,9}/`.

## 1. Eval round + tooling (Aug 8)

**Built** (all with fixture tests, green): `scripts/plot_by_subject.py`
(per-target-subject figures for teacher + curriculum CSVs), plus refreshed
`plot_teacher_evals` / `plot_curriculum_evals` outputs.

**Anomalies found in the eval set:**
- `src9_xf_aug` @66.6k was evaluated with **source sub2, not sub9** — its
  numbers are a cross-source read; teacher effectively never properly evaluated.
  TODO: `SOURCES=sub9 sh scripts/eval_one.sh src9_xf_aug`.
- One crashed pair in `ist_neutral_bn_mlp` (sub9←sub1, empty metrics) — plots
  now drop such rows loudly instead of averaging zeros.

## 2. Does training to completion matter? (matched-body checkpoint pairs)

| run | steps | success (same bodies) |
|---|---|---|
| src2_xf_aug | 16.2k → 61.2k | 60.3 → 80.8 (**+20.5**) |
| src2_xf_aug_retarget | 16.2k → 24.6k | 63.8 → 73.1 (+9.3) |
| src2_aug (MLP) | 16.2k → 52.5k | 83.2 → 87.2 (+3.9) |

**Result:** transformers need ~4× the training to approach the MLP; the MLP is
near ceiling by 16k. Retarget arms' weak *final* numbers were undertraining
artifacts (crash-restart sawtooth), not method verdicts.

## 3. Teacher standings + the distill decision (DEFERRED)

- MLP(aug): best mean at every epoch (82.5 @16.2k → 87.2 @52.5k).
- **sub16**: retarget_cpumotion 65.4% at only 16.2k; every non-retarget config
  ever trained (MLP stock, XF stock, XF nvadlr, XF nvadlr lowbuf @32.4k) sits
  at 2–38%. Retargeting is the only known fix.
- Matched-epoch, transformer-vs-transformer: retargeting +6 to +12 pts. The
  advantage is architecture-conditional so far — **the decisive cells are
  MLP + retargeting**, which is what gen-2 tests.
- Figure: `plots_by_subject/matched_epoch_success.png` (success vs epoch, all
  full-eval points; left = 16-shared-real-body mean, right = sub16 alone).
- Distill deferred until gen-2 answers whether retarget's sub16 fix composes
  with the MLP's mean. Clean follow-up if needed: src2-only distill A/B (MLP
  teacher vs retarget teacher, same student cfg).

## 4. PhysX OOM post-mortem → resume fixes (job 16502149)

`retarget_nvadlr` died at epoch 10,775 with PhysX narrowphase allocation
failures at 44.3/44G — **with lowbuf AND cpuMotionData already on**. Log
forensics: GPU-used crept 42.4 → 44.3G over 344k steps while torch-allocated
stayed flat at 12.5G → allocator-cache growth/fragmentation from the streamed
per-step motion gathers (memory PhysX cannot use). Fixes (commit 5bc78c9):
`PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256` + `NUM_ENVS` knob (2048 on 44G
cards / 4096 override). PSI ruled out: under cpuMotionData the PSI-tripled
`hoi_refs` lives in CPU RAM.

**Lowbuf verdict SETTLED (multiplier 20 → 12 is free):** same-arm reward curves
overlap exactly (normval_adlr vs _lowbuf: 10.4 vs 10.4 @0.22B frames, ~52 vs
~51.5 at end); contact reward at/above the multiplier-20 baseline (rcg 0.667
vs 0.618, all objects ≥). A 4-cell buf20 satellite in gen-2 re-tests this with
dedicated cells after the matched-epoch figure invited a misread.

## 5. Leak archaeology (the sub121 story generalized)

- **Two teacher generations**: runs finished before the Jul-21 sub121 drop
  (src2_aug@52.5k, src2_xf_aug@61.2k, src6/src9 finals) trained WITH sub121 →
  their sub13 cells are inflated. First clean sub13 number (new-gen
  normval_adlr_lowbuf @32.4k): **55.8** vs old-gen 82–88.
- **Hazard**: `smplx_teacher_src2_xf_aug/nn/` mixes both generations; the
  latest-checkpoint default grabs the contaminated 61.2k. Pass checkpoints
  explicitly for that arm.
- **Leak floor rule** (decision): threshold = smallest real-real subject
  distance = **2.106** (sub12↔sub13), computed from betas, not hardcoded — no
  training body may sit closer to a test body than two real humans ever are.
- Under the floor, **sub13 still leaks** in every arm ever trained: sub123
  @1.728, sub125 @1.957. sub10 (nearest 4.108) and sub16 (2.615) are clean, so
  the sub16 retargeting result stands.

## 6. K-fold body CV (is sub16 special, or is any OOD body hard?)

`scripts/generate_kfold_cfgs.py` (+ tests). fold0 = existing {sub10,13,16};
fold1 {sub5,7,12}; fold2 {sub8,15,17}; fold3 {sub1,3,14}. Never-test: sub4
(broken), sub9 (by request), sub2 (source), sub11/sub6 (nearest-source
near-identity). Folds dealt round-robin by beta-distance-to-source.
Note: the standalone kfold1–3 teachers are the *winner-CV* stage — do not run
alongside gen-2 (g2_mlp_plain_stock__f1 nearly duplicates kfold1).

## 7. The gen-2 grid (what's queued now) — 20 cells, 24h each

`scripts/generate_gen2_arms.py` (+ tests pinning every property). 16 main
cells `g2_{mlp,xf}_{plain,ret}_{stock,nvadlr}__f{0,1}` + 4-cell buffer
satellite `g2_{mlp,xf}_{plain,ret}_stock_buf20__f0`. Axes verified independent
by diffing the base files — the ONLY differing keys:

| axis | knobs |
|---|---|
| arch | numObs 3230/6524, useTransformerObs, network name |
| refs | retargetedMotionDir (+cpuMotionData, by necessity not swept) |
| recipe | normalize_value, lr_schedule (+kl_threshold 0.06) |
| fold | subjectBodies only |
| buffer (satellite) | default_buffer_size_multiplier 12 vs 20 |

**Design decisions (all Jess-approved):**
- **Uniform 2048 envs** in every cell — batch = envs×horizon must not differ
  between compared arms (option (a); `NUM_ENVS=4096` override for big GPUs).
- **Shared synthetic roster** (intersection-of-keeps across folds): both folds
  train on 13 reals + the SAME 30 synthetics = 43 bodies, so fold differences
  isolate the real-body swap. Cost = a uniform level shift, not a confound.
  CV teachers are instruments; the final teacher retrains on everything —
  BUT teacher OOD ability is also a deployment property (DAgger supervises
  unseen bodies), which is exactly why the eval keeps a moat.
- **Support-distance geometry measured, not equalized** (it can't be):
  f0 sub10@4.11 / sub13@2.11 / sub16@2.62; f1 sub5@2.72 / sub7@2.46 /
  sub12@2.11. Calibration: sub10 is the MOST isolated yet EASIEST body →
  support distance is a weak difficulty driver.
- Reward diagnostics (TERM_REASON / REWARD_BREAKDOWN / posechk) baked into
  every slurm script; guards fail-loud on any axis/leak-body mismatch.

Schedule: 4 cells/day → MLP f0, MLP f1, XF f0, XF f1, satellite. Eval per
cell: `HELDOUT="<fold trio>" sh scripts/eval_one.sh g2_<cell>`.

## 8. sub16 difficulty: three kinematic hypotheses down

`scripts/scan_reference_feasibility.py` (+ tests): FK sub2's dof+root on every
subject's skeleton, all 52 clips. **Floating is real but uncorrelated with
difficulty** — worst floaters sub10 (+10.5cm) and sub1 (+10.3) are the easiest
bodies; sub16 (+7.6) floats less than four easy ones. With beta-distance and
hand-reach (anti-correlated) already refuted, sub16's failure is
**dynamics-side** (balance/CoM/mass under sub2's joint angles). Next
discriminator: TERM_REASON tables (fell vs contact/ig death) — free in every
gen-2 log.

## 9. EgoExo4D basketball thread

Overfit policy on the CARI4D recon (sub100_bball_000.pt) exists and produced
rollouts; `slurm_cari4d_bball_overfit.sh` now rotation-ready (24h + debug
flags; cfg was already lowbuf @4096 envs — kept for resume continuity).
Next machinery: free-flight contact gating (dribble frames), then more clips
through cari4d_to_interact.

## 10. Open items

- Noise floor from repeat_A/repeat_B (local, free) — gates interpretation of
  small eval deltas.
- src9_xf_aug re-eval with SOURCES=sub9.
- SYNAUG_XF_POSE curriculum matrix missing body=sub17 row.
- Winner-CV (folds 2–3 for the gen-2 winner), then the distill A/B.
- Free-flight gating build; dormant: objectaug-physics, geomaug,
  transformer-curriculum follow-ups.
