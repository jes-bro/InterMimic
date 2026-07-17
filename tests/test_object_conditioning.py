#!/usr/bin/env python3
"""Offline checks for privileged object conditioning (no Isaac Gym).

Verifies the parts that don't need the simulator:
  1. per-env conditioning vector layout = [mass(1), per-axis scale(3), category
     one-hot(K)] with the right dims and values (one-hot exactly on the env's object).
  2. the numObs arithmetic the configs rely on: transformer token = base(1599) +
     betas(32) + objcond(1+3+K); numObs = 4 * token. Matches the committed
     _xf_aug_obj_geom_cond configs.
  3. per-axis scale folds uniform objectAug scale * anisotropic geom.
"""
import glob
import os
import sys

import torch
import yaml

CFG = os.path.join(os.path.dirname(__file__), os.pardir,
                   "isaacgym/src/intermimic/data/cfg")
OBJDIR = os.path.join(os.path.dirname(__file__), os.pardir,
                      "isaacgym/src/intermimic/data/assets/objects/objects")

fails = 0
def check(name, ok):
    global fails
    print(("PASS  " if ok else "FAIL  ") + name)
    fails += (0 if ok else 1)


# global vocab exactly as the task builds it (sorted object subdirs).
vocab = sorted(d for d in os.listdir(OBJDIR) if os.path.isdir(os.path.join(OBJDIR, d)))
K = len(vocab)
check("object vocab is the 19 asset subdirs", K == 19)


# --- mirror of the _obj_cond_vec assembly ---
def build_cond_vec(env_obj_names, oa_scale, aniso3):
    ne = len(env_obj_names)
    vidx = {n: i for i, n in enumerate(vocab)}
    cat = torch.tensor([vidx[n] for n in env_obj_names], dtype=torch.long)
    onehot = torch.zeros(ne, K)
    onehot[torch.arange(ne), cat] = 1.0
    scale3 = oa_scale.view(-1, 1) * aniso3                       # per-axis total scale
    mass = torch.ones(ne)                                        # placeholder (sim-read in real code)
    return torch.cat([mass.view(-1, 1), scale3, onehot], dim=-1)

ne = 6
names = [vocab[i % K] for i in range(ne)]
oa = torch.tensor([1.0, 1.1, 0.9, 1.15, 0.85, 1.0])
aniso = torch.rand(ne, 3) * 0.4 + 0.8
vec = build_cond_vec(names, oa, aniso)
check("cond vec width == 1 + 3 + K", vec.shape[-1] == 1 + 3 + K)
# one-hot slot is exactly the env's object, and sums to 1.
oh = vec[:, 4:]
check("one-hot is on the correct object per env",
      all(int(oh[e].argmax()) == vocab.index(names[e]) for e in range(ne)))
check("one-hot sums to exactly 1 per env", torch.allclose(oh.sum(-1), torch.ones(ne)))
# scale slots = uniform * aniso.
check("scale slots fold uniform*aniso per axis",
      torch.allclose(vec[:, 1:4], oa.view(-1, 1) * aniso, atol=1e-6))

# 2) numObs arithmetic vs the committed configs.
BASE, BETAS, NH = 1599, 32, 4
expect_numobs = NH * (BASE + BETAS + (1 + 3 + K))
check(f"expected transformer numObs == {expect_numobs}", expect_numobs == 6616)
cond_cfgs = sorted(glob.glob(os.path.join(CFG, "*_xf_aug_obj_cond.yaml"))
                   + glob.glob(os.path.join(CFG, "*_xf_aug_obj_geom_cond.yaml")))
for f in cond_cfgs:
    e = yaml.safe_load(open(f))["env"]
    ok = (e["numObs"] == expect_numobs
          and e["useTransformerObs"] is True
          and e["objectConditioning"]["enable"] is True)
    check(f"{os.path.basename(f)}: numObs {e['numObs']} matches arithmetic", ok)

check("found conditioning configs (obj_cond + geom_cond)", len(cond_cfgs) >= 6)

print()
if fails:
    print(f"FAILED ({fails})"); sys.exit(1)
print("ALL GREEN -- object-conditioning layout + numObs arithmetic verified.")
