#!/usr/bin/env python3
"""Tests for the freeFlightGate reward/reset split and the igRatio gating.

Source-level, in the pattern of tests/test_reward_breakdown_contact_split.py:
the task module imports isaacgym, which is not available off-cluster, so the
invariants are pinned by reading intermimic.py. What they protect is a set of
decisions that were MEASURED, and which a well-meaning edit could quietly undo:

  2026-09-02, r8's converged policy, one reset criterion at a time --
    all three off      100 steps, 100% success   (r8 as it ran)
    object only         23 steps,   0%           fatal
    igRatio only     96-99 steps,   0%           kills at the layup
    contactSteps only  100 steps, 100%           free: 0 diverges in 10240 eps

So: gate object, gate igRatio, do NOT gate contactSteps -- and keep the reward
half separable, because r8 trains without it.

Run:  python tests/test_free_flight_gate.py   (exit 0 = all green)
  or: pytest tests/test_free_flight_gate.py
"""
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(REPO, "isaacgym/src/intermimic/env/tasks/intermimic.py")
SRC = open(SRC_PATH).read()


def gate_block():
    """The body of `if self._free_flight_gate:` inside the REWARD path.

    The same guard also appears at config-parse time (for the startup print), so
    anchor on `ro_raw = ro`, which immediately precedes the reward-path one.
    """
    anchor = SRC.find("ro_raw = ro")
    assert anchor != -1, "the ro_raw anchor moved"
    i = SRC.find("if self._free_flight_gate:", anchor)
    assert i != -1, "the gate block moved or was renamed"
    j = SRC.find("# Shape the per-aspect factors", i)
    assert j != -1 and j > i, "could not find the end of the gate block"
    return SRC[i:j]


def test_reward_and_resets_are_separate_switches():
    assert "self._ffg_reward = bool(ffg_cfg.get('reward', _ffg_enable))" in SRC
    assert "self._ffg_resets = bool(ffg_cfg.get('resets', _ffg_enable))" in SRC
    # r8 trains without reward gating but needs the reset gating, so neither
    # half may imply the other
    assert "if self._ffg_reward:" in gate_block()
    assert "if self._ffg_resets:" in gate_block()
    print("ok: reward and reset halves are independent switches")


def test_legacy_enable_still_means_both():
    """Configs written before the split (the hoop_ffg arm) must not change
    meaning: enable: true = reward AND resets."""
    assert "_ffg_enable = bool(ffg_cfg.get('enable', False))" in SRC
    # both halves default to _ffg_enable when their own key is absent
    assert SRC.count("_ffg_enable))") == 2
    print("ok: legacy enable: true still means both halves")


def test_object_and_ig_resets_are_gated():
    block = gate_block()
    assert "object_reset = torch.logical_and(object_reset, held)" in block
    assert "ig_reset = torch.logical_and(ig_reset, held)" in block
    assert "held = ref_contact > 0.5" in block
    print("ok: object and igRatio resets are both gated on reference contact")


def test_contact_steps_is_NOT_gated():
    """contactSteps was measured free (100% completion ungated). Gating a
    criterion that never fires only discards signal on datasets where it would,
    so it must stay ungated."""
    block = gate_block()
    assert "contact_reset" not in block, (
        "contactSteps must not be gated -- it was measured free (100 steps, "
        "100% success, 0 contact_diverge in 10240 episodes)")
    print("ok: contactSteps is left ungated, as measured")


def test_the_gate_still_refuses_to_run_silently():
    """A gate that cannot read contact_obj does nothing at all; that must be an
    error, never a quiet no-op."""
    block = gate_block()
    assert "raise RuntimeError" in block
    assert "silently does nothing" in block
    print("ok: an unreadable contact_obj channel is an error, not a no-op")


def test_default_is_off_everywhere():
    """No config that omits the block may change behaviour."""
    assert "ffg_cfg.get('enable', False)" in SRC          # default False
    print("ok: the gate defaults off")


def test_unknown_gate_keys_are_rejected():
    m = re.search(r"badf = sorted\(k for k in \(rt\.get\('freeFlightGate'\) or \{\}\)\s*"
                  r"\n?\s*if k not in \(([^)]*)\)\)", SRC)
    assert m, "the freeFlightGate key validation moved"
    allowed = {s.strip().strip("'\"") for s in m.group(1).split(",") if s.strip()}
    assert allowed == {"enable", "reward", "resets"}, allowed
    print(f"ok: only {sorted(allowed)} accepted; a typo is an error")


def test_gate_probe_config_exists_and_is_minimal():
    """The eval cfg that tests this must differ from r8's by exactly the gate
    and the object threshold -- otherwise it measures something else."""
    import yaml
    base = os.path.join(REPO, "isaacgym/src/intermimic/data/cfg/"
                              "omomo_cari4d_bball_r8_horiz_eval.yaml")
    probe = os.path.join(REPO, "isaacgym/src/intermimic/data/cfg/"
                               "omomo_cari4d_bball_r8_gate_object_eval.yaml")
    if not (os.path.isfile(base) and os.path.isfile(probe)):
        print("skip: probe cfg not present")
        return
    b = yaml.safe_load(open(base))["env"]
    p = yaml.safe_load(open(probe))["env"]
    assert p["rewardTerms"]["freeFlightGate"]["enable"] is True
    assert p["resetThresholds"]["object"] == 0.5
    assert p["resetThresholds"]["igRatio"] is False
    assert p["resetThresholds"]["contactSteps"] is False
    assert p["resetThresholds"]["human"] == b["resetThresholds"]["human"]
    # rewardShape is deliberately ABSENT from the eval twins (eval stays on the
    # original product), so it is not in this list -- what matters is that the
    # probe inherits whatever the base does, checked by equality of key sets.
    for k in ("motion_file", "robotType", "rolloutLength", "obsHorizons", "numObs"):
        assert p[k] == b[k], k
    assert set(p) - set(b) == set(), set(p) - set(b)
    assert set(b) - set(p) == set(), set(b) - set(p)
    print("ok: the gate probe cfg differs only by the gate and object threshold")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\nall {len(fns)} tests passed")
