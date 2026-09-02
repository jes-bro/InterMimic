#!/usr/bin/env python3
"""Turn a bball-retarget run into a body-major reference tree, with a verdict
manifest and no silent drops.

TWO PROBLEMS THIS SOLVES.

1. Layout. slurm_cari4d_bball_retarget.sh writes one FLAT directory per target,

       InterAct/behave_cari4d_optj3d_cf2_<body>/<clip>.pt

   which is what the single-body arms (r12/r14) point motion_file at. The
   multi-body path wants BODY-MAJOR instead (intermimic.py:302),

       <retargetedMotionDir>/<body>/<clip>.pt

   and raises FileNotFoundError on a missing (body, clip) pair rather than
   falling back to the source reference.

2. Verdicts. retarget_contact.py exits 0 even when the solve made the reference
   WORSE, so exit status cannot be trusted; the numbers it prints are the only
   signal. Those numbers currently live in a job log that dies with the job.
   This writes them to a manifest next to the references, prints every body's
   status, and REFUSES to assemble a tree whose exclusions were not stated
   explicitly -- a dropped body must be an act, not a default.

    python3 scripts/retarget_layout.py --log retarget.out \\
        --flat-prefix InterAct/behave_cari4d_optj3d_cf2 \\
        --out-dir     InterAct/behave_cari4d_bball_f0 \\
        --bodies-from isaacgym/src/intermimic/data/cfg/omomo_teacher_g2_mlp_ret_stock__f0.yaml \\
        --dry-run

Statuses: PASS (contact error fell), FAILED (it did not), ERROR (the solve
started and printed no result), SKIP (no MJCF, never attempted). Only PASS is
usable; the retarget script's own guidance for FAILED is to raise --iters and
redo, and retrying beats dropping because every dropped body makes subjectBodies
differ from the arm this one is compared against.
"""
import argparse
import os
import re
import shutil
import sys

# "[retarget] sub100_bball_000.pt  sub100 -> sub14  object_scale=(1.0, 1.0, 1.0)"
RE_HEADER = re.compile(r"^\[retarget\]\s+(\S+)\s+(\S+)\s*->\s*(\S+)\s+object_scale=")
# "  contact err 4.81 -> 3.02 cm | all-body 6.10 -> 5.44 cm"
RE_ERRS = re.compile(r"^\s*contact err\s+([\d.]+)\s*->\s*([\d.]+)\s*cm"
                     r"(?:\s*\|\s*all-body\s+([\d.]+)\s*->\s*([\d.]+)\s*cm)?")
# "[bball-retarget] SKIP sub14: no MJCF at .../smplx_omomo_sub14.xml"
RE_SKIP = re.compile(r"SKIP\s+(\S+):\s*(.*)$")

USABLE = "PASS"
MANIFEST_NAME = "retarget_verdict.tsv"
MANIFEST_COLS = ["body", "status", "contact_before_cm", "contact_after_cm",
                 "all_before_cm", "all_after_cm", "clip", "source", "note"]


class LayoutError(Exception):
    """Refusal. The message is the whole explanation the user sees."""


def parse_log(text):
    """Retarget job log -> {body: verdict dict}, in the order encountered.

    Pairs each 'contact err' line with the most recent '[retarget]' header
    rather than zipping two independent lists: a solve that dies after its
    header would otherwise shift every later body's numbers onto the wrong body.
    """
    verdicts, pending = {}, None
    for line in text.splitlines():
        m = RE_SKIP.search(line)
        if m and "[retarget]" not in line:
            body, note = m.group(1), m.group(2).strip()
            verdicts[body] = dict(body=body, status="SKIP", clip="", source="",
                                  contact_before_cm=None, contact_after_cm=None,
                                  all_before_cm=None, all_after_cm=None, note=note)
            continue
        m = RE_HEADER.match(line)
        if m:
            clip, source, target = m.groups()
            # A header with no result line before the next header = a crashed solve.
            pending = dict(body=target, status="ERROR", clip=clip, source=source,
                           contact_before_cm=None, contact_after_cm=None,
                           all_before_cm=None, all_after_cm=None,
                           note="solve printed no contact-error line")
            verdicts[target] = pending
            continue
        m = RE_ERRS.match(line)
        if m and pending is not None:
            before, after = float(m.group(1)), float(m.group(2))
            pending["contact_before_cm"] = before
            pending["contact_after_cm"] = after
            if m.group(3) is not None:
                pending["all_before_cm"] = float(m.group(3))
                pending["all_after_cm"] = float(m.group(4))
            # The one criterion that matters: did the reference get better?
            pending["status"] = USABLE if after < before else "FAILED"
            pending["note"] = "" if after < before else "did not improve -- raise --iters and redo"
            pending = None
    return verdicts


def bodies_from_cfg(path):
    """Read subjectBodies out of an env YAML, so the body list has one source."""
    import yaml
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    bodies = (cfg.get("env") or {}).get("subjectBodies")
    if not bodies:
        raise LayoutError(f"{path} has no env.subjectBodies")
    return list(bodies)


def plan(verdicts, bodies, exclude=()):
    """-> (included, dropped, unaccounted).

    included    bodies to assemble (PASS, not excluded)
    dropped     [(body, status, why)] -- every body left out, with its reason
    unaccounted bodies with no verdict at all: the log never mentioned them, so
                we cannot say they passed OR failed. Always a refusal.
    """
    exclude = set(exclude)
    included, dropped, unaccounted = [], [], []
    for b in bodies:
        v = verdicts.get(b)
        if v is None:
            unaccounted.append(b)
            continue
        if b in exclude:
            dropped.append((b, v["status"], "excluded explicitly on the command line"))
        elif v["status"] == USABLE:
            included.append(b)
        else:
            dropped.append((b, v["status"], v["note"] or v["status"]))
    return included, dropped, unaccounted


def format_table(verdicts, bodies):
    """Every requested body, one line, whatever its status -- including bodies
    the log never mentioned. Printed on every run, not just failures."""
    lines = []
    for b in bodies:
        v = verdicts.get(b)
        if v is None:
            lines.append(f"  {b:>8}: {'(no verdict in log)':>28}   [UNACCOUNTED]")
            continue
        if v["contact_before_cm"] is None:
            nums = "-"
        else:
            nums = f"{v['contact_before_cm']:6.2f} -> {v['contact_after_cm']:6.2f} cm"
        note = f"  {v['note']}" if v["note"] else ""
        lines.append(f"  {b:>8}: {nums:>28}   [{v['status']}]{note}")
    return "\n".join(lines)


def write_manifest(path, verdicts, bodies):
    rows = ["\t".join(MANIFEST_COLS)]
    for b in bodies:
        v = verdicts.get(b) or dict(body=b, status="UNACCOUNTED", clip="", source="",
                                    contact_before_cm=None, contact_after_cm=None,
                                    all_before_cm=None, all_after_cm=None,
                                    note="log never mentioned this body")
        rows.append("\t".join("" if v.get(c) is None else str(v.get(c, ""))
                              for c in MANIFEST_COLS))
    with open(path, "w") as fh:
        fh.write("\n".join(rows) + "\n")


def assemble(flat_prefix, out_dir, bodies, clip_name, symlink=False):
    """Copy <flat_prefix>_<body>/<clip> -> <out_dir>/<body>/<clip>. Returns the
    written paths. A missing source file is an error, never a skip."""
    written = []
    for b in bodies:
        src = f"{flat_prefix}_{b}/{clip_name}"
        if not os.path.isfile(src):
            raise LayoutError(
                f"{b} is marked {USABLE} but its retargeted clip is missing at {src}. "
                f"The verdict and the files disagree -- not guessing which is right.")
        dst_dir = os.path.join(out_dir, b)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, clip_name)
        if os.path.lexists(dst):
            os.remove(dst)
        if symlink:
            os.symlink(os.path.abspath(src), dst)
        else:
            shutil.copy2(src, dst)
        written.append(dst)
    return written


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--log", required=True, help="retarget job log (the .out file)")
    p.add_argument("--flat-prefix", required=True,
                   help="OUT_PREFIX the retarget used; per-body dirs are <prefix>_<body>")
    p.add_argument("--out-dir", required=True, help="body-major tree to build")
    p.add_argument("--clip", default="sub100_bball_000.pt", help="clip filename")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--bodies-from", help="env YAML to read subjectBodies from")
    g.add_argument("--bodies", help="comma-separated body list")
    p.add_argument("--exclude", default="",
                   help="comma-separated bodies to drop ON PURPOSE. A non-PASS body "
                        "must be named here or the run refuses.")
    p.add_argument("--symlink", action="store_true", help="link instead of copy")
    p.add_argument("--dry-run", action="store_true",
                   help="print the table and the manifest that would be written")
    args = p.parse_args(argv)

    try:
        with open(args.log) as fh:
            verdicts = parse_log(fh.read())
        bodies = (bodies_from_cfg(args.bodies_from) if args.bodies_from
                  else [b.strip() for b in args.bodies.split(",") if b.strip()])
        exclude = [b.strip() for b in args.exclude.split(",") if b.strip()]
        included, dropped, unaccounted = plan(verdicts, bodies, exclude)
    except (OSError, LayoutError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"{len(bodies)} bodies requested, {len(verdicts)} have a verdict in {args.log}\n")
    print(format_table(verdicts, bodies))
    print(f"\n{len(included)} usable, {len(dropped)} dropped, {len(unaccounted)} unaccounted")

    if dropped:
        print("\nDROPPED -- each of these leaves subjectBodies different from the "
              "arm this one is compared against:")
        for b, status, why in dropped:
            print(f"  {b:>8}  [{status}]  {why}")

    if unaccounted:
        print(f"\nERROR: {len(unaccounted)} body(s) have no verdict in the log: "
              f"{', '.join(unaccounted)}", file=sys.stderr)
        print("  The log cannot say whether these passed or failed. Re-run the "
              "retarget for them, or drop them from the body list on purpose.",
              file=sys.stderr)
        return 1

    unstated = [(b, s) for b, s, _ in dropped if b not in exclude]
    if unstated:
        print(f"\nERROR: {len(unstated)} body(s) did not pass and were not named in "
              f"--exclude: {', '.join(b for b, _ in unstated)}", file=sys.stderr)
        print("  Retry them at higher --iters (preferred), or state the exclusion "
              "explicitly:", file=sys.stderr)
        print(f"    --exclude {','.join(b for b, _ in unstated)}", file=sys.stderr)
        return 1

    manifest = os.path.join(args.out_dir, MANIFEST_NAME)
    if args.dry_run:
        print(f"\n--dry-run: would write {manifest} and {len(included)} clip(s) "
              f"under {args.out_dir}/<body>/{args.clip}")
        return 0

    os.makedirs(args.out_dir, exist_ok=True)
    write_manifest(manifest, verdicts, bodies)
    try:
        written = assemble(args.flat_prefix, args.out_dir, included, args.clip,
                           symlink=args.symlink)
    except LayoutError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"\nwrote {manifest}")
    print(f"wrote {len(written)} clip(s) under {args.out_dir}/<body>/{args.clip}")
    print(f"\nset in the env cfg:\n  retargetedMotionDir: {args.out_dir}\n"
          f"  subjectBodies: {included}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
