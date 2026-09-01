#!/usr/bin/env python3
"""Front-end for the CARI4D -> InterMimic replay pipeline.

A collaborator has exactly one thing after a CARI4D optimisation run: the path
to a bundle,

    <CARI4D>/output/opt/cari4d-release+step031397_rectinj-hy3d3-optj3d/<seq>.pth

Everything else that scripts/slurm_cari4d_to_mimic.sh needs -- the matching
mesh, a free subject id, a dataset tag, a replay env YAML, the mandatory
gravity rotation -- is currently tribal knowledge, and four of those five fail
SILENTLY when wrong (wrong-build mesh, reused subject id clobbering another
body's MJCF, missing ROTATE_AXIS installing the clip upside down, and a gender
that retargets onto the other body model). This script derives or validates
each one from the bundle path, prints the whole resolved plan, and only then
submits.

    # see what it would do, write and submit nothing
    python3 scripts/cari4d_render.py --dry-run --gender male \\
        /simurgh2/projects/ret-hoi/CARI4D/output/opt/cari4d-release+step031397_rectinj-hy3d3-optj3d/<seq>.pth

    # do it
    python3 scripts/cari4d_render.py --gender male <bundle>.pth

Nothing here imports torch or isaacgym, so it is fast on a login node; the
heavy work stays in the sbatch job it submits.

Design rule throughout: when a value cannot be resolved unambiguously, FAIL and
print the candidates. Never pick one silently -- a wrong mesh or a reused
subject id produces a plausible-looking video of the wrong thing.
"""
import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# Cluster defaults, written inline rather than as $VARS so a copy-pasted
# command works as-is.
DEFAULT_CARI4D_ROOT = "/simurgh2/projects/ret-hoi/CARI4D"
DEFAULT_INTERMIMIC_ROOT = str(Path(__file__).resolve().parent.parent)

# The replay YAML every generated config is derived from. Chosen because it is
# the plain replay cfg (no staticScene, no training knobs) -- see --template to
# start from the hoop-in-scene one instead.
DEFAULT_TEMPLATE = "isaacgym/src/intermimic/data/cfg/omomo_cari4d_bball_rectinj3_replay.yaml"

# intermimic.py:63 parses a clip's subject as int(prefix[3:]), and sub0..sub17
# are the real OMOMO subjects. Converted CARI4D clips start above them.
FIRST_CARI4D_SUBJECT = 100


class ResolveError(Exception):
    """A value could not be resolved unambiguously. The message is the whole
    error the user sees, including the candidates they should choose between."""


# --------------------------------------------------------------------------
# Pure resolution logic (unit-tested in tests/test_cari4d_render.py)
# --------------------------------------------------------------------------

def seq_from_bundle(bundle: Path) -> str:
    """The CARI4D sequence name is the bundle's filename stem."""
    return Path(bundle).stem


def tag_from_release(release_dir: str) -> str:
    """Derive the InterAct dataset tag from the bundle's release directory.

    'cari4d-release+step031397_rectinj-hy3d3-optj3d'
        -> 'behave_cari4d_rectinj_hy3d3_optj3d'

    Derived rather than invented so two people converting from the same release
    land in the same InterAct directory. It must start with 'behave' or
    interact2mimic.py skips its SMPL-H / num_betas=10 branch -- which is what
    CARI4D bundles actually are.
    """
    # Strip the release+step prefix if present; what remains is the part that
    # actually distinguishes one optimisation config from another.
    distinctive = re.sub(r"^cari4d-release\+step\d+_", "", release_dir)
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", distinctive).strip("_").lower()
    if not slug:
        raise ResolveError(
            f"cannot derive a dataset tag from release directory {release_dir!r}; "
            f"pass --tag behave_cari4d_<something> explicitly")
    return f"behave_cari4d_{slug}"


def cfg_path_for_tag(tag: str) -> str:
    """Repo-relative path of the replay env YAML for a dataset tag.

    Relative on purpose: slurm_cari4d_to_mimic.sh tests "$INTERMIMIC/$CFG_ENV",
    so an absolute path there fails its existence check.
    """
    suffix = tag
    for prefix in ("behave_cari4d_", "behave_"):
        if suffix.startswith(prefix):
            suffix = suffix[len(prefix):]
            break
    return f"isaacgym/src/intermimic/data/cfg/omomo_cari4d_{suffix}_replay.yaml"


def find_mesh_candidates(cari4d_root: Path):
    """Every Hunyuan3D-aligned object mesh under the CARI4D data tree."""
    return sorted(Path(cari4d_root).glob("data/*/meshes-metric/*_align/*_align.obj"))


def resolve_mesh(cari4d_root: Path, seq: str):
    """Pick the object mesh belonging to `seq`, or raise listing the choices.

    Mesh directories are named '<seq-ish>_<NNN>_align', where the '<seq-ish>'
    part is not always the full sequence name (an optimisation run can append
    its own suffix to the bundle). So: try the full sequence name, then
    progressively shorter prefixes, and stop at the FIRST prefix length that
    matches anything. If that length matches more than one directory the
    sequence is genuinely ambiguous and the user must pass --mesh.

    Returns (path, matched_prefix) so the caller can show how it was matched --
    a prefix shorter than the sequence name is worth seeing.
    """
    candidates = find_mesh_candidates(cari4d_root)
    if not candidates:
        raise ResolveError(
            f"no meshes found under {cari4d_root}/data/*/meshes-metric/*_align/\n"
            f"  pass --mesh /path/to/<seq>_align.obj explicitly")

    # 4 chars is the shortest prefix worth trusting; below that every sequence
    # in a release matches and the "unique" answer would be an accident.
    for n in range(len(seq), 3, -1):
        prefix = seq[:n]
        hits = [c for c in candidates if c.parent.name.startswith(prefix)]
        if hits:
            if len(hits) == 1:
                return hits[0], prefix
            listed = "\n".join(f"    {h}" for h in hits)
            raise ResolveError(
                f"{len(hits)} meshes match sequence prefix {prefix!r} -- ambiguous, "
                f"not guessing:\n{listed}\n  pass --mesh <one of the above>")
    listed = "\n".join(f"    {c}" for c in candidates[:20])
    raise ResolveError(
        f"no mesh directory matches sequence {seq!r}. Candidates found:\n{listed}\n"
        f"  pass --mesh /path/to/<seq>_align.obj explicitly")


def installed_subjects(intermimic_root: Path):
    """Which sub<N> ids are already taken, and by what.

    Returns (by_tag, mjcf_ids):
      by_tag   {subject_id: {dataset_tag, ...}}  from InterAct/<tag>/sub<N>_*.pt
      mjcf_ids {subject_id, ...}                 from the installed per-subject
                                                 MJCFs, which carry no tag

    Both matter: step 3 of the pipeline writes
    assets/smplx/smplh_behave_sub<N>.xml, so reusing an id silently replaces a
    different person's body while leaving their motion file in place.
    """
    root = Path(intermimic_root)
    by_tag = {}
    for pt in root.glob("InterAct/*/sub*_*.pt"):
        m = re.match(r"^sub(\d+)_", pt.name)
        if m:
            by_tag.setdefault(int(m.group(1)), set()).add(pt.parent.name)
    mjcf_ids = set()
    for xml in root.glob("isaacgym/src/intermimic/data/assets/smplx/smplh_behave_sub*.xml"):
        m = re.search(r"sub(\d+)\.xml$", xml.name)
        if m:
            mjcf_ids.add(int(m.group(1)))
    return by_tag, mjcf_ids


def resolve_subject(by_tag, mjcf_ids, tag, requested=None):
    """Choose the sub<N> to install this clip as.

    Returns (subject_id, reason) where reason explains the choice in the plan.

    - no --subject-id: reuse the id this tag already owns (a re-run), else the
      lowest free id at or above 100.
    - explicit --subject-id: allowed if free or already owned by THIS tag;
      otherwise an error naming the tag that owns it.
    """
    owned_by_tag = sorted(sid for sid, tags in by_tag.items() if tag in tags)
    taken = set(by_tag) | set(mjcf_ids)

    if requested is None:
        if owned_by_tag:
            sid = owned_by_tag[0]
            return sid, f"re-run: {tag} already installed as sub{sid}"
        sid = FIRST_CARI4D_SUBJECT
        while sid in taken:
            sid += 1
        if sid == FIRST_CARI4D_SUBJECT:
            return sid, "first free id"
        blockers = ", ".join(
            f"sub{i}={'/'.join(sorted(by_tag.get(i, {'mjcf only'})))}"
            for i in sorted(taken) if FIRST_CARI4D_SUBJECT <= i < sid)
        return sid, f"first free id ({blockers})"

    if requested in taken and tag not in by_tag.get(requested, set()):
        owner = "/".join(sorted(by_tag.get(requested, set()))) or "an installed MJCF"
        free = FIRST_CARI4D_SUBJECT
        while free in taken:
            free += 1
        raise ResolveError(
            f"sub{requested} is already taken by {owner}. Installing over it would "
            f"replace that body's MJCF while leaving its motion file in place.\n"
            f"  use --subject-id {free} (first free), or omit --subject-id to "
            f"pick automatically")
    return requested, "requested explicitly"


def render_cfg(template_text: str, tag: str, subject_id: int, provenance: str) -> str:
    """Rewrite a replay env YAML for a new dataset tag + subject.

    Only three keys carry per-clip identity; each must appear exactly once, or
    the template is not what we think it is and we stop rather than write a
    config that half-points at the old clip.
    """
    edits = {
        "dataSub": (r"^(\s*dataSub:\s*).*$", rf"\g<1>['sub{subject_id}']"),
        "motion_file": (r"^(\s*motion_file:\s*).*$", rf"\g<1>InterAct/{tag}"),
        "robotType": (r"^(\s*robotType:\s*).*$",
                      rf'\g<1>"smplx/smplh_behave_sub{subject_id}.xml"'),
    }
    text = template_text
    for key, (pattern, repl) in edits.items():
        text, n = re.subn(pattern, repl, text, flags=re.MULTILINE)
        if n != 1:
            raise ResolveError(
                f"template has {n} '{key}:' lines, expected exactly 1 -- "
                f"refusing to generate a config from it")
    return provenance + text


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Resolve and submit a CARI4D bundle -> InterMimic replay.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Design rule")[0])
    p.add_argument("bundle", help="CARI4D bundle .pth (output/opt/<release>/<seq>.pth)")
    p.add_argument("--gender", required=True, choices=["male", "female", "neutral"],
                   help="SMPL-H gender CARI4D reconstructed this sequence with. "
                        "Required and never guessed: a wrong value does not error, "
                        "it retargets onto the other body model and reads as a bad "
                        "retarget.")
    p.add_argument("--mesh", help="object mesh .obj (default: resolved from the "
                                  "sequence name under <cari4d-root>/data)")
    p.add_argument("--tag", help="InterAct dataset tag (default: derived from the "
                                 "bundle's release directory)")
    p.add_argument("--subject-id", type=int, help="install as sub<N> (default: reuse "
                                                  "this tag's id, else first free)")
    p.add_argument("--object-name", default="bball")
    p.add_argument("--clip-idx", default="000")
    p.add_argument("--rotate-axis", default="x",
                   help="gravity-alignment axis passed to rotate_pt (default x -- "
                        "without it the clip installs upside down, silently)")
    p.add_argument("--rotate-degrees", default="180")
    p.add_argument("--no-rotate", action="store_true",
                   help="install unrotated (you almost never want this)")
    p.add_argument("--template", default=DEFAULT_TEMPLATE,
                   help=f"replay YAML to derive the env config from "
                        f"(default {DEFAULT_TEMPLATE})")
    p.add_argument("--cari4d-root", default=DEFAULT_CARI4D_ROOT)
    p.add_argument("--intermimic-root", default=DEFAULT_INTERMIMIC_ROOT)
    p.add_argument("--dry-run", action="store_true",
                   help="print the resolved plan; write nothing, submit nothing")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    mimic_root = Path(args.intermimic_root).resolve()

    bundle = Path(args.bundle)
    if not bundle.is_file():
        print(f"ERROR: no bundle at {bundle}", file=sys.stderr)
        return 1
    bundle = bundle.resolve()
    seq = seq_from_bundle(bundle)

    try:
        tag = args.tag or tag_from_release(bundle.parent.name)
        if not tag.startswith("behave"):
            raise ResolveError(
                f"dataset tag {tag!r} must start with 'behave' or interact2mimic.py "
                f"skips the SMPL-H branch CARI4D bundles need")

        if args.mesh:
            mesh, mesh_note = Path(args.mesh).resolve(), "given with --mesh"
            if not mesh.is_file():
                raise ResolveError(f"no mesh at {mesh}")
        else:
            mesh, prefix = resolve_mesh(Path(args.cari4d_root), seq)
            mesh_note = (f"matched on prefix {prefix!r}"
                         + ("" if prefix == seq else
                            f" -- SHORTER than the sequence name, check this is the "
                            f"right build"))

        by_tag, mjcf_ids = installed_subjects(mimic_root)
        subject_id, subject_note = resolve_subject(by_tag, mjcf_ids, tag,
                                                   args.subject_id)
    except ResolveError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    cfg_rel = cfg_path_for_tag(tag)
    cfg_abs = mimic_root / cfg_rel
    cfg_exists = cfg_abs.is_file()

    rotate = None if args.no_rotate else (args.rotate_axis, args.rotate_degrees)

    print("resolved:")
    print(f"  seq         {seq}")
    print(f"  bundle      {bundle}")
    print(f"  mesh        {mesh}")
    print(f"              ({mesh_note})")
    print(f"  gender      {args.gender}")
    print(f"  subject     sub{subject_id}   ({subject_note})")
    print(f"  tag         {tag}")
    print(f"  object/clip {args.object_name} / {args.clip_idx}")
    if rotate:
        print(f"  rotate      axis={rotate[0]} degrees={rotate[1]} drop-to-floor=1")
    else:
        print("  rotate      DISABLED (--no-rotate)")
    print(f"  cfg_env     {cfg_rel}")
    print(f"              ({'exists, reused as-is' if cfg_exists else 'missing, will be generated from ' + args.template})")

    # Generate the replay config when absent. Never overwrite one: an existing
    # config may have been hand-edited (a staticScene hoop block, different
    # numEnvs) and silently rewriting someone's config is worse than failing.
    if not cfg_exists:
        template = mimic_root / args.template
        if not template.is_file():
            print(f"ERROR: no template config at {template}", file=sys.stderr)
            return 1
        provenance = (
            f"# GENERATED by scripts/cari4d_render.py on "
            f"{datetime.now().strftime('%Y-%m-%d')}\n"
            f"#   from template: {args.template}\n"
            f"#   for bundle:    {bundle}\n"
            f"# Safe to hand-edit; this script never overwrites an existing config.\n")
        try:
            text = render_cfg(template.read_text(), tag, subject_id, provenance)
        except ResolveError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        if args.dry_run:
            print(f"\n  would write {cfg_rel} ({len(text.splitlines())} lines)")
        else:
            cfg_abs.write_text(text)
            print(f"\n  wrote {cfg_rel}")

    env = dict(os.environ)
    env.update({
        "BUNDLE": str(bundle),
        "MESH": str(mesh),
        "SUBJECT_ID": str(subject_id),
        "OBJECT_NAME": args.object_name,
        "CLIP_IDX": args.clip_idx,
        "GENDER": args.gender,
        "DATASET_TAG": tag,
        "CFG_ENV": cfg_rel,
    })
    if rotate:
        env["ROTATE_AXIS"], env["ROTATE_DEGREES"] = rotate

    # Print the equivalent shell command so the submission is auditable and can
    # be re-run by hand without this script.
    shown = ["BUNDLE", "MESH", "SUBJECT_ID", "OBJECT_NAME", "CLIP_IDX", "GENDER",
             "DATASET_TAG", "CFG_ENV", "ROTATE_AXIS", "ROTATE_DEGREES"]
    print("\nsubmitting:")
    for k in shown:
        if k in env and (k not in os.environ or env[k] != os.environ.get(k)):
            print(f"  {k}={env[k]} \\")
    print("  sbatch scripts/slurm_cari4d_to_mimic.sh")

    if args.dry_run:
        print("\nnot submitted (--dry-run)")
        return 0

    return subprocess.call(["sbatch", "scripts/slurm_cari4d_to_mimic.sh"],
                           cwd=str(mimic_root), env=env)


if __name__ == "__main__":
    sys.exit(main())
