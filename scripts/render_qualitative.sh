#!/bin/bash
# Generate qualitative comparison videos: source motion + 4 retargets to sub10.
#
# Outputs 5 MP4s in /tmp/:
#   render_source_sub2.mp4         — sub2 body playing sub2's own motion (kinematic, no policy)
#   render_full_method.mp4         — full method (both_normreward) retargeting to sub10
#   render_reward_ablation.mp4     — both (no body-norm reward) retargeting to sub10
#   render_betas_ablation.mp4      — both_nobetas_normreward retargeting to sub10
#   render_vanilla.mp4             — InterMimic vanilla baseline on sub10
#
# All 5 use the SAME motion clip (maxClipsPerObject=1 + dataSub/dataObjects
# constraints) and SAME initial frame (stateInit="Start") for direct comparison.
#
# Run from repo root on a machine with a GPU + xvfb (or skip xvfb if display works).

set -e
cd "$(dirname "$0")/.."

CFG_DIR=isaacgym/src/intermimic/data/cfg
TRAIN_DIR=isaacgym/src/intermimic/data/cfg/train/rlg
TARGET_BODY=sub10
SOURCE_SUB=sub2
OBJECT=largetable
MAX_FRAMES=300

# --- create 3 render yamls (3230 obs, 3198 obs, source) ---

# 3230-dim obs yaml (for full method + reward ablation, sub10 body)
cat > "$CFG_DIR/omomo_render_target_3230.yaml" <<EOF
env:
  numEnvs: 1
  envSpacing: 2
  episodeLength: 300
  isFlagrun: False
  enableDebugVis: False
  playdataset: False
  projtype: "None"
  saveImages: False
  initVel: False
  moreRigid: False
  pdControl: True
  powerScale: 1.0
  controlFrequencyInv: 2
  stateInit: "Start"
  rolloutLength: 300
  hybridInitProb: 0.0
  dataFPS: 30
  dataFramesScale: 1
  dataSub: ['$SOURCE_SUB']
  subjectBodies: ['$TARGET_BODY']
  dataObjects: ['$OBJECT']
  maxClipsPerObject: 1
  betas_file: scripts/omomo_betas.npz
  bodyNormalizedReward: True
  ballSize: 1.
  numObs: 3230
  numObsRetarget: 3230
  useTransformerObs: False
  motion_file: InterAct/OMOMO_new
  motion_file_retarget: InterAct/OMOMO_new
  robotType: "smplx/omomo.xml"
  objectDensity: 200
  localRootObs: False
  keyBodies: ["L_Hip", "L_Knee", "L_Ankle", "L_Toe", "R_Hip", "R_Knee", "R_Ankle", "R_Toe", "Torso", "Spine", "Chest", "Neck", "Head", "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist"]
  contactBodies: ["L_Hip", "L_Knee", "L_Ankle", "L_Toe", "R_Hip", "R_Knee", "R_Ankle", "R_Toe", "Torso", "Spine", "Chest", "Neck", "Head", "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "L_Index3", "L_Middle3", "L_Pinky3", "L_Ring3","L_Thumb3","R_Index3", "R_Middle3", "R_Pinky3", "R_Ring3","R_Thumb3"]
  terminationHeight: 0.15
  enableEarlyTermination: True
  physicalBufferSize: 3
  teacherPolicy: checkpoints/teachers/crosspair_both
  teacherPolicyCFG: intermimic/data/cfg/train/rlg/omomo_crosspair_b10_s2_largetable.yaml

  asset:
    assetRoot: "intermimic/data/assets"
  plane:
    staticFriction: 0.9
    dynamicFriction: 0.9
    restitution: 0.7
  rewardWeights:
    p: 30.
    r: 2.5
    pv: 0.
    rv: 0.
    op: 5.0
    or: 0.1
    opv: 0.1
    orv: 0.
    ig: 5.
    cg_hand: 5.
    cg_other: 5.
    cg_all: 3.
    eg1: 0.00002
    eg2: 0.00002
    eg3: 0.000000001

sim:
  substeps: 2
  physx:
    num_threads: 4
    solver_type: 1
    num_position_iterations: 4
    num_velocity_iterations: 1
    contact_offset: 0.02
    rest_offset: 0.0
    bounce_threshold_velocity: 0.2
    max_depenetration_velocity: 100.0
    default_buffer_size_multiplier: 20.0
    max_gpu_contact_pairs: 34603008
  flex:
    num_inner_iterations: 10
    warm_start: 0.25
EOF

# 3198-dim obs yaml (for betas ablation + vanilla, sub10 body)
sed 's/numObs: 3230/numObs: 3198/; s/numObsRetarget: 3230/numObsRetarget: 3198/; s/betas_file: scripts\/omomo_betas.npz//' \
    "$CFG_DIR/omomo_render_target_3230.yaml" > "$CFG_DIR/omomo_render_target_3198.yaml"

# Source replay yaml (sub2 body, kinematic playback, doesn't matter much for obs dim)
sed "s/subjectBodies: \\['$TARGET_BODY'\\]/subjectBodies: ['$SOURCE_SUB']/" \
    "$CFG_DIR/omomo_render_target_3230.yaml" > "$CFG_DIR/omomo_render_source.yaml"

echo "=== render yamls created ==="
ls -l "$CFG_DIR/omomo_render_*.yaml"

# --- render commands ---

# 1. Source replay (kinematic, sub2 body doing sub2 motion)
echo ""
echo "=== Render 1/5: Source motion (sub2 kinematic) ==="
RECORD_VIDEO=/tmp/render_source_${SOURCE_SUB}.mp4 MAX_VIDEO_FRAMES=$MAX_FRAMES \
python -u -m intermimic.run_distill \
    --task InterMimic_CrossPair \
    --cfg_env "$CFG_DIR/omomo_render_source.yaml" \
    --cfg_train "$TRAIN_DIR/omomo_distill_both_normreward.yaml" \
    --play_dataset --headless --num_envs 1

# 2. Full method on sub10
echo ""
echo "=== Render 2/5: Full method (both_normreward) on $TARGET_BODY ==="
RECORD_VIDEO=/tmp/render_full_method.mp4 MAX_VIDEO_FRAMES=$MAX_FRAMES \
python -u -m intermimic.run_distill \
    --task InterMimic_CrossPair \
    --cfg_env "$CFG_DIR/omomo_render_target_3230.yaml" \
    --cfg_train "$TRAIN_DIR/omomo_distill_both_normreward.yaml" \
    --play --checkpoint output/smplx_distill_both_normreward/nn/mimic.pth \
    --headless --num_envs 1

# 3. Reward ablation (both) on sub10
echo ""
echo "=== Render 3/5: Reward ablation (both) on $TARGET_BODY ==="
RECORD_VIDEO=/tmp/render_reward_ablation.mp4 MAX_VIDEO_FRAMES=$MAX_FRAMES \
python -u -m intermimic.run_distill \
    --task InterMimic_CrossPair \
    --cfg_env "$CFG_DIR/omomo_render_target_3230.yaml" \
    --cfg_train "$TRAIN_DIR/omomo_distill_both.yaml" \
    --play --checkpoint output/smplx_distill_both/nn/mimic.pth \
    --headless --num_envs 1

# 4. Betas ablation (both_nobetas_normreward) on sub10 — uses 3198 obs
echo ""
echo "=== Render 4/5: Betas ablation (both_nobetas_normreward) on $TARGET_BODY ==="
RECORD_VIDEO=/tmp/render_betas_ablation.mp4 MAX_VIDEO_FRAMES=$MAX_FRAMES \
python -u -m intermimic.run_distill \
    --task InterMimic_CrossPair \
    --cfg_env "$CFG_DIR/omomo_render_target_3198.yaml" \
    --cfg_train "$TRAIN_DIR/omomo_distill_both_nobetas_normreward.yaml" \
    --play --checkpoint output/smplx_distill_both_nobetas_normreward/nn/mimic.pth \
    --headless --num_envs 1

# 5. Vanilla InterMimic baseline on sub10 — uses 3198 obs, task=InterMimic (not crosspair)
echo ""
echo "=== Render 5/5: Vanilla InterMimic on $TARGET_BODY ==="
RECORD_VIDEO=/tmp/render_vanilla.mp4 MAX_VIDEO_FRAMES=$MAX_FRAMES \
python -u -m intermimic.run \
    --task InterMimic \
    --cfg_env "$CFG_DIR/omomo_render_target_3198.yaml" \
    --cfg_train "$TRAIN_DIR/omomo_multibody_nobetas.yaml" \
    --play --checkpoint checkpoints/student.pth \
    --headless --num_envs 1

echo ""
echo "=== All renders done. Videos in /tmp/ ==="
ls -lh /tmp/render_*.mp4
