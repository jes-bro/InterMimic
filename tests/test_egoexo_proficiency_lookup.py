"""egoexo_proficiency_lookup.py on fixture takes/annotations: the join is on
take_uid only, both annotation layouts are accepted, unlabelled takes say so.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_egoexo_proficiency_lookup.py -q
"""
import importlib.util
import json
import os

import pytest

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "egoexo_proficiency_lookup.py")
spec = importlib.util.spec_from_file_location("egoexo_proficiency_lookup", SCRIPT)
pl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pl)

TAKES = [
    {"take_name": "sfu_basketball_07_40", "take_uid": "u-novice", "task_name": "Basketball Drills - Reverse Layup",
     "parent_task_name": "Basketball", "university_name": "sfu", "participant_uid": 424, "duration_sec": 21.77},
    {"take_name": "unc_basketball_03-31-23_02_9", "take_uid": "u-expert", "task_name": "Basketball Drills - Reverse Layup",
     "parent_task_name": "Basketball", "university_name": "unc", "participant_uid": 387, "duration_sec": 64.07},
    {"take_name": "nus_cpr_01_1", "take_uid": "u-cpr", "task_name": "CPR", "parent_task_name": "Health",
     "university_name": "nus", "participant_uid": 900, "duration_sec": 100.0},
]


@pytest.fixture
def fx(tmp_path):
    json.dump(TAKES, open(tmp_path / "takes.json", "w"))
    ann = tmp_path / "ann"; ann.mkdir()
    # list layout, wrapped in {'ds', 'annotations'} like the real files
    json.dump({"ds": "egoexo", "annotations": [
        {"take_uid": "u-novice", "origin_participant_id": 999, "proficiency_score": "Novice"}]},
        open(ann / "proficiency_demonstrator_train.json", "w"))
    # dict layout, also accepted
    json.dump({"annotations": {"u-expert": {"proficiency_score": "Late Expert"}}},
              open(ann / "proficiency_demonstrator_val.json", "w"))
    # commentary file: must be ignored (no tier in it)
    json.dump({"annotations": [{"take_uid": "u-cpr", "commentary": "good"}]},
              open(ann / "proficiency_demonstration_train.json", "w"))
    return str(tmp_path / "takes.json"), str(ann)


def test_tiers_join_on_take_uid(fx):
    takes, ann = fx
    rows = pl.main(["sfu_basketball_07_40", "--takes", takes, "--ann-dir", ann])
    assert rows[0]["tier"] == "Novice" and rows[0]["participant"] == 424
    rows = pl.main(["--uid", "u-expert", "--takes", takes, "--ann-dir", ann])
    assert rows[0]["tier"] == "Late Expert"
    rows = pl.main(["nus_cpr_01_1", "--takes", takes, "--ann-dir", ann])
    assert rows[0]["tier"] == "no label"                       # CPR has none; commentary file ignored


def test_grep_and_participant(fx):
    takes, ann = fx
    assert len(pl.main(["--grep", "basketball", "--takes", takes, "--ann-dir", ann])) == 2
    assert pl.main(["--participant", "387", "--takes", takes, "--ann-dir", ann])[0]["take_name"] == "unc_basketball_03-31-23_02_9"
    with pytest.raises(SystemExit, match="no matching take"):
        pl.main(["nope", "--takes", takes, "--ann-dir", ann])
