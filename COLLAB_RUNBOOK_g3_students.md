# Running the g3 students on another machine

Branch `g3-distill`. Everything the students need -- teacher checkpoints, the
13 contact-retargeted OMOMO trees, OMOMO clips, the three activity datasets,
object assets and all body MJCFs -- is in the GCS bucket `gs://jesb-intermimic`.
You need read access (Jess grants it), `gcloud` installed and logged in, an
Isaac Gym + torch env that runs this repo, a C++ compiler on PATH (`torch.compile`
needs it for the transformer), and ~120 GB of disk.

    gcloud auth login                                # YOUR Google account (Jess grants it objectViewer on the
                                                     # bucket; no project membership or `config set project` needed)
    git fetch origin g3-distill && git checkout g3-distill
    export INTERMIMIC_ENV=<your conda env name>     # the launchers activate this
    export NUM_ENVS=1024                             # ALL students run at 1024 envs (40 GB cards); keep it for comparability

All commands from the repo root. Each step refuses loudly if something is
missing -- read the error, don't work around it.

## OMOMO student (13 per-source teachers -> one student)

    sh scripts/gcp_stage.sh pull-assets            # 43+ body MJCFs
    sh scripts/gcp_stage.sh pull-omomo-data        # ~60 GB: OMOMO_new clips + 13 retarget trees
    python3 scripts/merge_retarget_trees.py --sources InterAct/OMOMO_retarget_contact_src1 InterAct/OMOMO_retarget_contact_src2 InterAct/OMOMO_retarget_contact_src3 InterAct/OMOMO_retarget_contact_src5 InterAct/OMOMO_retarget_contact_src6 InterAct/OMOMO_retarget_contact_src7 InterAct/OMOMO_retarget_contact_src8 InterAct/OMOMO_retarget_contact_src9 InterAct/OMOMO_retarget_contact_src11 InterAct/OMOMO_retarget_contact_src12 InterAct/OMOMO_retarget_contact_src14 InterAct/OMOMO_retarget_contact_src15 InterAct/OMOMO_retarget_contact_src17 --out InterAct/OMOMO_retarget_contact_srcall13 --bodies-from isaacgym/src/intermimic/data/cfg/omomo_student_g3_omomo_xf_ret_nvadlr__f0.yaml
    sh scripts/gcp_stage.sh pull-teachers smplx_teacher_g3_omomo_geoall__f0 smplx_teacher_g3_omomo_geoall_src1__f0 smplx_teacher_g3_omomo_geoall_src3__f0 smplx_teacher_g3_omomo_geoall_src5__f0 smplx_teacher_g3_omomo_geoall_src6__f0 smplx_teacher_g3_omomo_geoall_src7__f0 smplx_teacher_g3_omomo_geoall_src8__f0 smplx_teacher_g3_omomo_geoall_src9__f0 smplx_teacher_g3_omomo_geoall_src11__f0 smplx_teacher_g3_omomo_geoall_src12__f0 smplx_teacher_g3_omomo_geoall_src14__f0 smplx_teacher_g3_omomo_geoall_src15__f0 smplx_teacher_g3_omomo_geoall_src17__f0
    python3 scripts/collect_g3_teachers.py --omomo-sources 1 2 3 5 6 7 8 9 11 12 14 15 17 --out checkpoints/teachers/g3_omomo
    # (needs torch: reads each checkpoint's epoch; prints 13 lines, writes teachers.yaml)

    sh scripts/gcp_run_in_tmux.sh slurm_student_g3_omomo_mlp_ret_stock__f0.sh omomo_mlp    # MLP
    sh scripts/gcp_run_in_tmux.sh slurm_student_g3_omomo_xf_ret_nvadlr__f0.sh  omomo_xf     # transformer

RAM: this data set is ~226 GB resident (ragged storage, streamed from CPU); the
machine needs ~300 GB. The reference load takes 10-20 minutes with nothing printed.

## OMOMO student on the FINAL teachers (`_tfinal`)

The OMOMO XF student above was distilled from the teachers as they stood on
2026-09-16, mid-training. `_tfinal` is the same student on the teachers' final
checkpoints. Branch `g3-distill-tfinal`. The only differences: its own name,
and its own teacher dir so the set it learned from is recorded on disk.

    git fetch origin g3-distill-tfinal && git checkout g3-distill-tfinal
    sh scripts/gcp_stage.sh pull-teachers smplx_teacher_g3_omomo_geoall_src1__f0 smplx_teacher_g3_omomo_geoall_src2__f0 smplx_teacher_g3_omomo_geoall_src3__f0 smplx_teacher_g3_omomo_geoall_src5__f0 smplx_teacher_g3_omomo_geoall_src6__f0 smplx_teacher_g3_omomo_geoall_src7__f0 smplx_teacher_g3_omomo_geoall_src8__f0 smplx_teacher_g3_omomo_geoall_src9__f0 smplx_teacher_g3_omomo_geoall_src11__f0 smplx_teacher_g3_omomo_geoall_src12__f0 smplx_teacher_g3_omomo_geoall_src14__f0 smplx_teacher_g3_omomo_geoall_src15__f0 smplx_teacher_g3_omomo_geoall_src17__f0
    python3 scripts/collect_g3_teachers.py --omomo-sources 1 2 3 5 6 7 8 9 11 12 14 15 17 --out checkpoints/teachers/g3_omomo_tfinal
    NUM_ENVS=1024 sh scripts/gcp_run_in_tmux.sh slurm_student_g3_omomo_xf_ret_nvadlr_tfinal__f0.sh omomo_xf_tfinal

Before the pull: the teachers in the bucket must be the FINAL ones. Each
teacher's last snapshot is pushed from the machine that trained it with
`sh scripts/gcp_stage.sh push-teacher <exp>`; the collect step prints the
epoch it picked for every source -- check those are the final epochs, not
44-67k. Data is the same pull as the OMOMO student above (skip if present).
NUM_ENVS=1024 matches the Sep-16 run. Push `mimic_00004000.pth` and
`mimic_00009000.pth` to the bucket when they land
(`gcloud storage cp checkpoints/smplx_student_g3_omomo_xf_ret_nvadlr_tfinal__f0/nn/mimic_0000N000.pth gs://jesb-intermimic/checkpoints/smplx_student_g3_omomo_xf_ret_nvadlr_tfinal__f0/nn/`).

## Activity student on the FINAL teachers (`_tfinal`)

Same idea as the OMOMO `_tfinal` above, for the activity student: the Sep-16 run
used the three activity teachers mid-training. Branch `g3-distill-tfinal`.

    git fetch origin g3-distill-tfinal && git checkout g3-distill-tfinal
    sh scripts/gcp_stage.sh pull-teachers smplx_teacher_g3_bball7_geoall__f0 smplx_teacher_g3_soccer15_geoall__f0 smplx_teacher_g3_cpr13_geoall__f0
    python3 scripts/collect_g3_teachers.py --activities bball7 soccer15 cpr13 --out checkpoints/teachers/g3_act_tfinal
    NUM_ENVS=1024 sh scripts/gcp_run_in_tmux.sh slurm_student_g3_act_xf_ret_nvadlr_tfinal__f0.sh act_xf_tfinal

The data pull is the same as the activity student above (skip if present); only
the teachers are re-pulled. The collect step prints the epoch it picked per
teacher -- RECORD those three numbers, they are what "final" means for this run.
cpr13 may still be training, in which case its number is the latest snapshot at
launch, not its wall. NUM_ENVS=1024 matches the Sep-16 run; on a VM that already
had a tmux server running, `tmux kill-server` first or the new session inherits
the old environment and NUM_ENVS silently reverts to the launcher default.

## `nopose` students (no relative joint-angle reward factor), branch `g3-distill-nopose`

Two arms with `rewardTerms.pose.enable: false`; everything else is the tfinal
recipe. This is a METHOD CANDIDATE, not an ablation: the teacher nopose arm beat
the with-pose teacher, so the pose term is being dropped from the method and
these arms test that on the students. In a student the env reward only reaches the loss through the critic
(from epoch 2300) and the PPO actor term (after 2900), so read both at a
MATCHED EPOCH well past 2900, never at matched wall time.

OMOMO XF on the final teachers, one key off `_tfinal` (same teacher dir, same data):

    git fetch origin g3-distill-nopose && git checkout g3-distill-nopose
    sbatch --exclude=simurgh6,simurgh2 slurm_student_g3_omomo_xf_ret_nvadlr_tfinal_nopose__f0.sh

Activities WITHOUT cpr (bball7 + soccer15, 22 sources): a NEW source set, so its
own merged data, props file and teacher dir first (each step refuses to redo):

    python3 scripts/merge_activity_data.py --arms bball7 soccer15 --out-motion InterAct/behave_cari4d_act_nocpr --out-retarget InterAct/behave_cari4d_act_nocpr_f0_bodymajor --props-out isaacgym/src/intermimic/data/cfg/object_props_g3_act_nocpr.yaml --bodies-from isaacgym/src/intermimic/data/cfg/omomo_student_g3_act_nocpr_xf_ret_nvadlr_nopose__f0.yaml --student-plane-restitution 0.7
    python3 scripts/collect_g3_teachers.py --activities bball7 soccer15 --out checkpoints/teachers/g3_act_nocpr
    sbatch --exclude=simurgh6,simurgh2 slurm_student_g3_act_nocpr_xf_ret_nvadlr_nopose__f0.sh

Commit the props file the merge writes (`object_props_g3_act_nocpr.yaml`) next
to the cfg. The collect step prints the epoch it picked per teacher; record the
two numbers. No with-pose twin of this source set exists, so its read against
the act students confounds "no cpr" with "no pose".

## Activity student (bball7 + soccer15 + cpr13 teachers -> one student)

    sh scripts/gcp_stage.sh pull-assets
    sh scripts/gcp_stage.sh pull-act-data          # three motion dirs + retarget trees + assets/objects
    sh scripts/gcp_stage.sh pull-teachers smplx_teacher_g3_bball7_geoall__f0 smplx_teacher_g3_soccer15_geoall__f0 smplx_teacher_g3_cpr13_geoall__f0
    python3 scripts/merge_activity_data.py --arms bball7 soccer15 cpr13 --out-motion InterAct/behave_cari4d_act --out-retarget InterAct/behave_cari4d_act_f0_bodymajor --props-out isaacgym/src/intermimic/data/cfg/object_props_g3_act.yaml --bodies-from isaacgym/src/intermimic/data/cfg/omomo_student_g3_act_xf_ret_nvadlr__f0.yaml --student-plane-restitution 0.7
    python3 scripts/collect_g3_teachers.py --activities bball7 soccer15 cpr13 --out checkpoints/teachers/g3_act

    sh scripts/gcp_run_in_tmux.sh slurm_student_g3_act_mlp_ret_stock__f0.sh act_mlp
    sh scripts/gcp_run_in_tmux.sh slurm_student_g3_act_xf_ret_nvadlr__f0.sh  act_xf

Small: ~45 GB RAM, loads in a couple of minutes.

## Watching a run

`gcp_run_in_tmux.sh` runs the launcher ONCE in a detached tmux and tees the
output to `<launcher>-gcp.log` (no automatic restart). `tail -f` that log:

    [student] teachers: N files; missing sources: none
    [distill-g3] teacher obs 9594 over horizons [1, 4, 7, 10, 13, 16]; student obs 9594 ...
    [distill-g3] teacher subN.pth: sources [N], epoch E ...     (one per teacher)
    [distill-g3] N teachers cover M sources
    [object] <name>: PhysX mass ... kg (target ...)           (activity only, one per object)
    epoch_num:1 ...

`tmux attach -t <name>` to see it live (Ctrl-b then d to detach; never Ctrl-c).
Checkpoints land in `checkpoints/smplx_student_g3_<arm>__f0/nn/`;
`checkpoints/teachers/<set>/teachers.yaml` records which teacher epoch the
student saw. If a run exits, the log's last line says so; rerunning the same
launcher resumes from its latest checkpoint, never starts fresh over it.

## What is where already (2026-09-16), so nothing runs twice

| student | XF | MLP |
|---|---|---|
| omomo (13 per-source) | GCP `student-omomo` | not started |
| act (bball7+soccer15+cpr13) | simurgh | not started |
| omomo_halves, act7 | simurgh / GCP | not started |
