"""clip_object_flight_stats.py: the airborne share and the candidate exponent on
synthetic 591-channel clips with known object behaviour.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_clip_object_flight_stats.py -q
"""
import importlib.util
import os

import numpy as np
import pytest
import torch

REPO = os.path.join(os.path.dirname(__file__), "..")
spec = importlib.util.spec_from_file_location("cof", os.path.join(REPO, "scripts", "clip_object_flight_stats.py"))
cof = importlib.util.module_from_spec(spec); spec.loader.exec_module(cof)


def _clip(T, obj_z, contact):
    """(T, 591) with the object's z and the contact_obj channel set; everything else 0."""
    x = np.zeros((T, 591))
    x[:, cof.I_OBJ_Z] = obj_z
    x[:, cof.I_CONTACT_OBJ] = contact
    return x


def test_held_object_is_never_airborne():
    # OMOMO: carried at chest height the whole clip, hand contact on -> held, not flying
    s = cof.flight_stats(_clip(120, np.full(120, 1.2), np.ones(120)))
    assert s["free_share"] == 0 and s["airborne_share"] == 0 and s["air_run_s"] == 0
    assert s["e_air"] == 1.0 and s["e_flight"] == 1.0 and s["flight_m"] == 0


def test_resting_object_with_no_contact_is_not_airborne():
    # OMOMO approach phase: object on the floor, nobody touching it
    s = cof.flight_stats(_clip(90, np.full(90, 0.3), np.zeros(90)))
    assert s["free_share"] == 1.0 and s["airborne_share"] == 0 and s["e_air"] == 1.0 and s["e_flight"] == 1.0


def test_rolling_ball_is_free_but_grounded():
    # soccer: ball rolls along the ground (constant z, tiny jitter), touched 10% of frames
    z = 0.11 + 0.01 * np.sin(np.linspace(0, 20, 300))
    contact = (np.arange(300) % 10 == 0).astype(float)
    s = cof.flight_stats(_clip(300, z, contact))
    assert s["free_share"] == pytest.approx(0.9)
    assert s["airborne_share"] == 0 and s["flight_m"] == 0 and s["e_flight"] == 1.0


def test_dribble_is_mostly_airborne_and_reaches_the_root():
    # basketball dribble: the ball bounces between floor (z=0.12) and hand (z=1.0) every 20
    # frames; hand contact only at the top 2 frames of each bounce
    T = 300
    phase = (np.arange(T) % 20) / 20.0
    z = 0.12 + 0.88 * np.sin(np.pi * phase)          # 0 at floor, 1 at hand
    contact = (np.abs(phase - 0.5) < 0.06).astype(float)
    s = cof.flight_stats(_clip(T, z, contact))
    assert s["airborne_share"] >= 0.6                  # in the air most of the time
    assert 0.15 <= s["air_run_s"] <= 0.4               # one bounce: ~6 airborne frames of 20
    assert s["e_air"] == pytest.approx(1 / cof.N_FACTORS)
    assert 0.4 < s["air_height_m"] < 0.8 and s["flight_m"] > 0.3     # ~0.6 share x ~0.55 m mean height
    assert s["e_flight"] < 0.5
    assert 0.7 < s["max_height"] < 0.88                # peak above the 5th-percentile "ground" (a little above the floor)


def test_candidate_exponent_shape():
    assert cof.candidate_exponent(0.0, 0.6) == 1.0 and cof.candidate_exponent(0.0, 0.5) == 1.0
    assert cof.candidate_exponent(0.3, 0.6) == pytest.approx(0.5)
    assert cof.candidate_exponent(0.6, 0.6) == pytest.approx(0.25)
    assert cof.candidate_exponent(0.9, 0.6) == 0.25            # never below 1/N


def test_collect_filters_and_reads_pt_files(tmp_path):
    def save(name, z, c):
        x = torch.tensor(_clip(60, np.full(60, z), np.full(60, c)), dtype=torch.float32)
        torch.save(x, tmp_path / f"{name}.pt")
    save("sub401_ball_000", 1.0, 0.0)       # airborne (free, high above its own floor? no: constant z -> ground = z)
    save("sub2_woodchair_000", 0.5, 1.0)
    rows = cof.collect(str(tmp_path), sources=["sub2"])
    assert [r["clip"] for r in rows] == ["sub2_woodchair_000"] and rows[0]["object"] == "woodchair"
    # a constant-height free object is its own floor: never airborne
    assert cof.collect(str(tmp_path), objects=["ball"])[0]["airborne_share"] == 0
