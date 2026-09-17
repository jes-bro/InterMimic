#!/usr/bin/env python3
"""Who is an EgoExo4D take, and how skilled? Joins takes.json with the
proficiency_demonstrator annotations on take_uid.

    python3 scripts/egoexo_proficiency_lookup.py sfu_basketball_07_40
    python3 scripts/egoexo_proficiency_lookup.py --uid 3eb43914-2546-4e08-ab04-9283ac66806e
    python3 scripts/egoexo_proficiency_lookup.py --grep unc_basketball_03-31-23_02   # every take of a capture
    python3 scripts/egoexo_proficiency_lookup.py --participant 424               # every take of a person

Prints take_name, task, university, participant, duration, and the tier
(Novice / Early Expert / Intermediate Expert / Late Expert) or 'no label'.
CPR (Health) has no proficiency labels at all -- expect 'no label' there.

Defaults: ~/egoexo4d/takes.json and ~/Downloads/annotations-egoexo4d/
proficiency_demonstrator_{train,val}.json; override with --takes / --ann-dir.
The annotation record's `origin_participant_id` is NOT the take's
`participant_uid`; the join is on take_uid only.
"""
import argparse
import json
import os
import sys

HOME = os.path.expanduser("~")
TAKES = os.path.join(HOME, "egoexo4d", "takes.json")
ANN_DIR = os.path.join(HOME, "Downloads", "annotations-egoexo4d")


def load_takes(path):
    with open(path) as fh:
        takes = json.load(fh)
    if not isinstance(takes, list):
        raise SystemExit(f"ERROR: {path}: expected a list of takes")
    return takes


def load_tiers(ann_dir):
    """{take_uid: tier} from every proficiency_demonstrator_*.json in ann_dir."""
    tiers = {}
    files = sorted(f for f in os.listdir(ann_dir) if f.startswith("proficiency_demonstrator") and f.endswith(".json"))
    if not files:
        raise SystemExit(f"ERROR: no proficiency_demonstrator_*.json in {ann_dir}")
    for f in files:
        with open(os.path.join(ann_dir, f)) as fh:
            doc = json.load(fh)
        anns = doc.get("annotations", doc) if isinstance(doc, dict) else doc
        # the file is {take_uid: {...proficiency_score...}} or a list of records
        items = anns.items() if isinstance(anns, dict) else ((r.get("take_uid"), r) for r in anns)
        for uid, rec in items:
            tier = rec.get("proficiency_score") if isinstance(rec, dict) else None
            if uid and tier:
                tiers[uid] = tier
    return tiers


def describe(take, tiers):
    return {
        "take_name": take.get("take_name"), "take_uid": take.get("take_uid"),
        "task": take.get("task_name"), "parent_task": take.get("parent_task_name"),
        "university": take.get("university_name"), "participant": take.get("participant_uid"),
        "duration_s": round(float(take.get("duration_sec") or 0), 1),
        "tier": tiers.get(take.get("take_uid"), "no label"),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take_name", nargs="?", help="exact take_name")
    ap.add_argument("--uid", help="exact take_uid")
    ap.add_argument("--grep", help="substring of take_name (a capture prefix, say)")
    ap.add_argument("--participant", help="participant_uid: list all their takes")
    ap.add_argument("--takes", default=TAKES)
    ap.add_argument("--ann-dir", default=ANN_DIR)
    a = ap.parse_args(argv)
    if not any([a.take_name, a.uid, a.grep, a.participant]):
        ap.error("give a take_name, --uid, --grep or --participant")

    takes, tiers = load_takes(a.takes), load_tiers(a.ann_dir)
    if a.take_name:
        hits = [t for t in takes if t.get("take_name") == a.take_name]
    elif a.uid:
        hits = [t for t in takes if t.get("take_uid") == a.uid]
    elif a.grep:
        hits = [t for t in takes if a.grep in (t.get("take_name") or "")]
    else:
        hits = [t for t in takes if str(t.get("participant_uid")) == str(a.participant)]
    if not hits:
        raise SystemExit("no matching take (names are like unc_basketball_03-31-23_02_9)")
    rows = [describe(t, tiers) for t in hits]
    w = max(len(r["take_name"]) for r in rows)
    for r in sorted(rows, key=lambda r: r["take_name"]):
        print(f"{r['take_name']:<{w}}  {r['tier']:<20}  {r['task']}  | {r['university']} participant {r['participant']}"
              f"  {r['duration_s']}s  uid {r['take_uid']}")
    return rows


if __name__ == "__main__":
    main()
