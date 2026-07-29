#!/usr/bin/env python3
"""Tests for the throughput sweep tooling: cfg generation and log summarising.

The failure modes worth pinning are the SILENT ones -- a probe that runs happily
but measures the wrong thing, or a table that omits a crashed probe so the
configuration looks untested rather than infeasible.
"""
import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

import make_probe_cfg  # noqa: E402
import summarize_throughput  # noqa: E402


@pytest.fixture
def base_cfgs(tmp_path):
    """A minimal stand-in for the retarget arm's cfg pair."""
    env = {
        "env": {"numEnvs": 4096, "retargetedMotionDir": "InterAct/x", "physicalBufferSize": 3},
        "sim": {"physx": {"default_buffer_size_multiplier": 20.0,
                          "max_gpu_contact_pairs": 34603008}},
    }
    train = {"params": {"config": {"max_epochs": 100000, "save_best_after": 100,
                                   "full_experiment_name": "smplx_teacher_real"}}}
    p_env, p_train = tmp_path / "env.yaml", tmp_path / "train.yaml"
    p_env.write_text(yaml.safe_dump(env))
    p_train.write_text(yaml.safe_dump(train))
    return str(p_env), str(p_train)


# ---------------------------------------------------------------- cfg building

def test_control_probe_changes_nothing_but_epochs_and_name(base_cfgs):
    """A control probe must be the real arm, except it stops early."""
    be, bt = base_cfgs
    env, train, applied = make_probe_cfg.build(be, bt, "control", 20)
    assert applied == []
    assert env == yaml.safe_load(open(be)), "control probe perturbed the env cfg"
    assert train["params"]["config"]["max_epochs"] == 20
    assert train["params"]["config"]["full_experiment_name"] == "probe_control"


def test_knobs_land_in_the_right_section(base_cfgs):
    be, bt = base_cfgs
    env, _, applied = make_probe_cfg.build(
        be, bt, "t", 20, num_envs=6144, cpu_motion=True,
        buffer_mult=10.0, contact_pairs=16777216)
    assert env["env"]["numEnvs"] == 6144
    assert env["env"]["cpuMotionData"] is True
    assert env["sim"]["physx"]["default_buffer_size_multiplier"] == 10.0
    assert env["sim"]["physx"]["max_gpu_contact_pairs"] == 16777216
    assert len(applied) == 4


def test_probe_can_never_write_a_checkpoint(base_cfgs):
    """epochs >= save_best_after would leave checkpoints behind -- refuse."""
    be, bt = base_cfgs
    with pytest.raises(ValueError, match="save_best_after"):
        make_probe_cfg.build(be, bt, "t", 100)


def test_experiment_name_is_always_namespaced(base_cfgs):
    """A probe must never inherit a real run's checkpoint directory."""
    be, bt = base_cfgs
    _, train, _ = make_probe_cfg.build(be, bt, "cpumotion", 20)
    name = train["params"]["config"]["full_experiment_name"]
    assert name.startswith("probe_") and "smplx_teacher_real" not in name


def test_unknown_env_knob_fails_loudly(base_cfgs):
    """Silently adding a key the task ignores would measure nothing."""
    be, bt = base_cfgs
    env = yaml.safe_load(open(be))
    del env["env"]["numEnvs"]
    open(be, "w").write(yaml.safe_dump(env))
    with pytest.raises(KeyError, match="numEnvs"):
        make_probe_cfg.build(be, bt, "t", 20, num_envs=6144)


# ------------------------------------------------------------- log summarising

def _log(tag, epochs, mem=True, host="simurgh5", envs=4096, extra=""):
    lines = [f"[probe] tag={tag} host={host} job=1", f"num_envs: {envs}"]
    if mem:
        lines.append("[mem] motion tensors: hoi_data 4.32G (2704, 354, 1211) + "
                     "hoi_refs 3.55G (2704, 3, 354, 332) = 7.87G on GPU")
        lines.append("[mem] step 201: torch 20.42G | GPU used 43.5/44G")
        lines.append("[mem] step 401: torch 20.42G | GPU used 43.7/44G")
    for e, (s, t) in enumerate(epochs, start=1):
        lines.append(f"epoch_num:{e} mean_rewards:[0.16] fps step: {s} fps total: {t}")
    lines.append(extra)
    return "\n".join(lines)


def test_warmup_epochs_are_dropped(tmp_path):
    """Early epochs read high; including them would flatter every probe."""
    # epochs 1-5 are wildly fast, 6-10 are the truth.
    eps = [(99999.0, 99999.0)] * 5 + [(7000.0, 5000.0)] * 5
    p = tmp_path / "a.log"
    p.write_text(_log("control", eps))
    r = summarize_throughput.parse(str(p), warmup=5)
    assert r["fps_step"] == 7000.0 and r["fps_total"] == 5000.0
    assert r["n_epochs"] == 5


def test_median_resists_one_contended_epoch(tmp_path):
    eps = [(0.0, 0.0)] * 5 + [(7000.0, 5000.0), (7000.0, 5000.0), (10.0, 10.0)]
    p = tmp_path / "a.log"
    p.write_text(_log("control", eps))
    assert summarize_throughput.parse(str(p), warmup=5)["fps_step"] == 7000.0


def test_mem_and_motion_location_parsed(tmp_path):
    p = tmp_path / "a.log"
    p.write_text(_log("control", [(1.0, 1.0)] * 10))
    r = summarize_throughput.parse(str(p), warmup=5)
    assert r["peak_used"] == 43.7 and r["total_mem"] == 44.0
    assert r["motion_size"] == 7.87 and r["motion_where"] == "on GPU"
    assert r["num_envs"] == 4096


def test_oom_probe_is_reported_not_dropped(tmp_path):
    """A missing row reads as 'not run'; it actually means 'does not fit'."""
    p = tmp_path / "a.log"
    p.write_text(_log("env8192", [], extra="RuntimeError: CUDA out of memory"))
    r = summarize_throughput.parse(str(p), warmup=5)
    assert r["failure"] == "OOM"
    lines = summarize_throughput.render([r])
    assert any("FAILED: OOM" in ln for ln in lines)


def test_table_sorts_best_first_and_flags_mixed_hosts(tmp_path):
    fast, slow = tmp_path / "f.log", tmp_path / "s.log"
    fast.write_text(_log("cpumotion", [(0.0, 0.0)] * 5 + [(9000.0, 7000.0)] * 3,
                         host="simurgh5"))
    slow.write_text(_log("control", [(0.0, 0.0)] * 5 + [(7000.0, 5000.0)] * 3,
                         host="simurgh7"))
    rows = [summarize_throughput.parse(str(p)) for p in (slow, fast)]
    lines = summarize_throughput.render(rows, ref_step=7250.0, ref_total=5450.0)
    body = [ln for ln in lines if ln.startswith(("cpumotion", "control"))]
    assert body[0].startswith("cpumotion"), "table must sort fastest first"
    assert any("DIFFERENT hosts" in ln for ln in lines), "mixed hosts must warn"
    assert any("+28" in ln for ln in body), "delta vs reference missing"


def test_all_warmup_is_a_failure_not_a_zero(tmp_path):
    """A probe that died at epoch 3 must not report as 0 fps."""
    p = tmp_path / "a.log"
    p.write_text(_log("control", [(7000.0, 5000.0)] * 3))
    r = summarize_throughput.parse(str(p), warmup=5)
    assert r["failure"] and "WARMUP" in r["failure"].upper()
