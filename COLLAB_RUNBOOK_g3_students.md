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
