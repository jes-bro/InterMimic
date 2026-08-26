#!/usr/bin/env python3
"""Pin the ref-contact split in the REWARD_BREAKDOWN diagnostic.

WHY THIS EXISTS. r3_roll30's dribble regressed against r2_warm ("kicks the ball
away"), and the hypothesis was that the object reward `ro` is being spent on
free-flight frames the policy cannot affect. The clip-averaged breakdown could
not test that: r2 and r3 visit different frames, so their averages are not
comparable at all (r2 only ever ran frames 0..~38). Splitting the SAME run's
terms by whether the REFERENCE says hand-object contact is comparable, because
both rows come from one policy on one distribution.

The reading the split supports:
  large free-vs-held gap in ro -> reward is being wasted on ballistic frames,
                                  which is the case for rewardTerms.freeFlightGate
  uniformly mediocre ro        -> the gate will NOT help; look elsewhere

Two properties must hold or the diagnostic lies:
  A. the bincount arithmetic actually partitions by the flag   -> test_split_math
  B. the rows report the UNGATED ro                            -> test_ungated
B is the subtle one: with the gate ON, the gated ro is 1.0 by construction on
every 'free' frame, so logging it would make the diagnostic agree with itself
and always "confirm" the gate was needed.

Isaac Gym cannot be imported here, so the accumulation is reproduced against the
same torch ops the task uses rather than by instantiating the task.

Run:  python tests/test_reward_breakdown_contact_split.py   (exit 0 = all green)
"""
import os
import re
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASK = os.path.join(REPO, "isaacgym/src/intermimic/env/tasks/intermimic.py")

failures = []


def check(label, condition, detail=""):
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label} {detail}")
        failures.append(label)


def accumulate(rb, ro, rig, rcg, ref_contact):
    """Reproduce the ref-contact accumulation from _log_reward_breakdown."""
    T = torch.stack([rb, ro, rig, rcg, rb * ro * rig * rcg], dim=1)
    ids = (ref_contact > 0.5).long()
    cnts = torch.bincount(ids, minlength=2).float()
    sums = torch.zeros((2, 5))
    for k in range(5):
        sums[:, k] += torch.bincount(ids, weights=T[:, k], minlength=2)
    return sums, cnts


def test_split_math():
    print("1. the split partitions by the reference contact flag:")
    # 3 free envs with a poor object reward, 2 held envs with a good one --
    # the signature the hypothesis predicts.
    ro = torch.tensor([0.10, 0.10, 0.10, 0.90, 0.90])
    rb = torch.full((5,), 0.5)
    rig = torch.full((5,), 0.5)
    rcg = torch.full((5,), 0.5)
    flag = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0])

    sums, cnts = accumulate(rb, ro, rig, rcg, flag)
    check("counts split 3 free / 2 held", cnts.tolist() == [3.0, 2.0],
          f"(got {cnts.tolist()})")
    free_ro, held_ro = (sums[0, 1] / cnts[0]).item(), (sums[1, 1] / cnts[1]).item()
    check("free-row ro is the free envs' mean", abs(free_ro - 0.10) < 1e-6,
          f"(got {free_ro})")
    check("held-row ro is the held envs' mean", abs(held_ro - 0.90) < 1e-6,
          f"(got {held_ro})")
    check("the gap this diagnostic exists to surface is visible",
          held_ro - free_ro > 0.5)

    # A run with NO free frames must not produce a bogus zero row -- the print
    # path skips rows with count 0, and a fabricated 0.0 would read as "ro
    # collapsed in flight" when there was no flight at all.
    sums, cnts = accumulate(rb, ro, rig, rcg, torch.ones(5))
    check("all-held run leaves the free row empty, not zero", cnts[0].item() == 0.0,
          f"(got count {cnts[0].item()})")

    # The uniform case must NOT look like the gate's case.
    flat = torch.full((5,), 0.4)
    sums, cnts = accumulate(rb, flat, rig, rcg, flag)
    gap = (sums[1, 1] / cnts[1] - sums[0, 1] / cnts[0]).item()
    check("uniform ro shows ~no gap (gate would not help)", abs(gap) < 1e-6,
          f"(gap {gap})")


def test_ungated():
    """With the gate ON, gated ro is 1.0 on every free frame by construction.
    The split must report the RAW value or it cannot measure anything."""
    print("\n2. the split reports the UNGATED ro:")
    ro_raw = torch.tensor([0.10, 0.10, 0.90, 0.90])
    flag = torch.tensor([0.0, 0.0, 1.0, 1.0])
    ro_gated = ro_raw * flag + (1.0 - flag)      # the task's gate expression
    check("gate neutralizes free frames to 1.0, leaves held frames alone",
          torch.allclose(ro_gated, torch.tensor([1.0, 1.0, 0.9, 0.9]), atol=1e-6),
          f"(got {ro_gated.tolist()})")

    rest = torch.full((4,), 0.5)
    raw_sums, cnts = accumulate(rest, ro_raw, rest, rest, flag)
    gated_sums, _ = accumulate(rest, ro_gated, rest, rest, flag)
    check("logging the RAW ro preserves the free-row signal (0.10)",
          abs((raw_sums[0, 1] / cnts[0]).item() - 0.10) < 1e-6)
    check("logging the GATED ro would erase it (1.00) -- the bug this avoids",
          abs((gated_sums[0, 1] / cnts[0]).item() - 1.00) < 1e-6)


def test_source_wiring():
    """The math above is only real if the task actually wires it that way."""
    print("\n3. task source wiring:")
    src = open(TASK).read()

    check("'ref-contact' is a breakdown group", "'ref-contact'" in src)
    check("the split passes ro_raw, not the post-gate ro",
          re.search(r"ro_d = ro_raw if ro_raw is not None else ro", src) is not None)
    # ro_raw must be captured BEFORE the gate rewrites ro, or it is the same value.
    i_raw = src.find("ro_raw = ro")
    i_gate = src.find("ro = ro * ref_contact + (1.0 - ref_contact)")
    check("ro_raw is captured before the gate rewrites ro",
          i_raw != -1 and i_gate != -1 and i_raw < i_gate,
          f"(raw at {i_raw}, gate at {i_gate})")
    # The flag must be read whether or not the gate is on, else the split only
    # works on runs that already enabled the thing it is meant to justify.
    i_try = src.find("ref_contact = self.extract_data_component(")
    i_if = src.find("if self._free_flight_gate:", i_try)
    check("the contact flag is read unconditionally (not inside the gate branch)",
          i_try != -1 and i_if != -1 and i_try < i_if)
    # A missing channel must cost only the diagnostic...
    check("a missing contact channel degrades to ref_contact=None",
          "ref_contact = None" in src)
    # ...but must HARD FAIL if the gate is on, per the no-silent-fallbacks rule:
    # a gate that silently does nothing would look like a completed experiment.
    check("an enabled gate with no channel raises instead of no-op'ing",
          "freeFlightGate is enabled but the contact_obj" in src)
    check("empty groups say so rather than printing a bare heading",
          "(no data -- channel unavailable)" in src)
    check("the header states the gate's state so rows are unambiguous",
          "freeFlightGate is %s" in src)


def main():
    test_split_math()
    test_ungated()
    test_source_wiring()
    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("all green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
