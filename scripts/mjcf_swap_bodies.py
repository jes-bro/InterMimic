#!/usr/bin/env python3
"""Build MJCF variants for bisecting which NUMBERS of a per-subject body trip the
simulator (the sub4 investigation): the per-subject files are structurally
identical, so the fault must live in geometry numbers, and swapping a body
subtree between two subjects localises it.

  # sub4 with sub2's whole left leg (L_Hip subtree: hip, knee, ankle, toe)
  python3 scripts/mjcf_swap_bodies.py --base sub4 --donor sub2 --bodies L_Hip \
      --out isaacgym/src/intermimic/data/assets/smplx/smplx_variant_sub4_leftleg_from_sub2.xml
  # sub2 with sub4's left leg
  python3 scripts/mjcf_swap_bodies.py --base sub2 --donor sub4 --bodies L_Hip \
      --out isaacgym/src/intermimic/data/assets/smplx/smplx_variant_sub2_leftleg_from_sub4.xml
  # perturb one capsule radius instead of swapping
  python3 scripts/mjcf_swap_bodies.py --base sub4 --scale-geom L_Knee:1.01 \
      --out isaacgym/src/intermimic/data/assets/smplx/smplx_variant_sub4_knee_r101.xml

--bodies swaps the ENTIRE subtree rooted at each named body (its pos, geom,
joints and all descendants) from the donor into the base. The base body's
'pos' (its offset in the parent) is taken from the donor as well, so the
swapped leg is the donor's leg exactly. Replay a variant with
REPLAY_ASSET_FILE=smplx/<file>.xml (humanoid.py), subject name unchanged.
The variant is written next to the per-subject files; those are gitignored,
so `git add -f` it if it must travel to the cluster.
"""
import argparse
import copy
import xml.etree.ElementTree as ET
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[1] / "isaacgym/src/intermimic/data/assets/smplx"


def find_body(root, name):
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "body" and child.get("name") == name:
                return parent, child
    raise SystemExit(f"ERROR: body {name!r} not found")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="subject whose file is the starting point, e.g. sub4")
    ap.add_argument("--donor", help="subject whose bodies replace the base's, e.g. sub2")
    ap.add_argument("--bodies", nargs="*", default=[], help="root body names of the subtrees to swap")
    ap.add_argument("--scale-geom", nargs="*", default=[],
                    help="<body>:<factor> scale that body's geom radius (size[0]) by factor")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    base_tree = ET.parse(ASSETS / f"smplx_omomo_{a.base}.xml")
    base = base_tree.getroot()
    n_body0 = sum(1 for _ in base.iter("body")); n_joint0 = sum(1 for _ in base.iter("joint"))
    if a.bodies:
        if not a.donor:
            raise SystemExit("ERROR: --bodies needs --donor")
        donor = ET.parse(ASSETS / f"smplx_omomo_{a.donor}.xml").getroot()
        for name in a.bodies:
            parent, old = find_body(base, name)
            _, new = find_body(donor, name)
            idx = list(parent).index(old)
            parent.remove(old)
            parent.insert(idx, copy.deepcopy(new))
            print(f"[swap] {name}: subtree replaced with {a.donor}'s "
                  f"({sum(1 for _ in new.iter('body'))} bodies)")
    for item in a.scale_geom:
        name, factor = item.split(":")
        _, body = find_body(base, name)
        g = body.find("geom")
        s = g.get("size").split()
        s[0] = f"{float(s[0]) * float(factor):.6f}"
        g.set("size", " ".join(s))
        print(f"[scale] {name}: geom size[0] x{factor} -> {s[0]}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    base_tree.write(a.out, encoding="unicode", xml_declaration=False)
    # sanity: same body/joint counts as the base file (153 hinges + the pelvis free joint)
    n_body = sum(1 for _ in base.iter("body")); n_joint = sum(1 for _ in base.iter("joint"))
    print(f"wrote {a.out}  ({n_body} bodies, {n_joint} joints)")
    if (n_body, n_joint) != (n_body0, n_joint0):
        raise SystemExit(f"ERROR: variant has {n_body} bodies / {n_joint} joints; base had {n_body0} / {n_joint0}")


if __name__ == "__main__":
    main()
