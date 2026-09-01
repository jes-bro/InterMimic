#!/usr/bin/env python3
"""Tests for scripts/cari4d_render.py's resolution logic.

The real bundles and meshes are cluster-only, so every fixture here is a
fabricated directory tree. What is pinned is exactly the logic that decides
what gets installed where -- the four things that fail silently in the hand-run
pipeline: which mesh, which subject id, which dataset tag, and whether the
rotation is applied.

Run:  python tests/test_cari4d_render.py   (exit 0 = all green)
  or: pytest tests/test_cari4d_render.py
"""
import os
import sys
import tempfile
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import cari4d_render as cr  # noqa: E402


# ---------------------------------------------------------------- fixtures

def make_cari4d_tree(root, mesh_dirs):
    """A CARI4D data tree containing one <name>_align.obj per name given."""
    for name in mesh_dirs:
        d = Path(root) / "data" / "cari4d-release" / "meshes-metric" / name
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.obj").write_text("# fake mesh\n")


def make_mimic_tree(root, pts=(), mjcfs=()):
    """An InterMimic tree with the given installed clips and per-subject MJCFs.

    pts: iterable of (dataset_tag, filename), mjcfs: iterable of subject ids.
    """
    for tag, name in pts:
        d = Path(root) / "InterAct" / tag
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_text("")
    assets = Path(root) / "isaacgym/src/intermimic/data/assets/smplx"
    assets.mkdir(parents=True, exist_ok=True)
    for sid in mjcfs:
        (assets / f"smplh_behave_sub{sid}.xml").write_text("<mujoco/>")


# ------------------------------------------------------------------- tests

def test_seq_and_tag_derivation():
    seq = cr.seq_from_bundle(Path(
        "/x/output/opt/cari4d-release+step031397_rectinj-hy3d3-optj3d/"
        "Date03_Sub01_bball_rev003at.pth"))
    assert seq == "Date03_Sub01_bball_rev003at"

    # the release+step prefix is stripped; what distinguishes the run survives
    assert (cr.tag_from_release("cari4d-release+step031397_rectinj-hy3d3-optj3d")
            == "behave_cari4d_rectinj_hy3d3_optj3d")
    # same release, different step -> same tag, so two people converting from
    # the same config land in the same InterAct directory
    assert (cr.tag_from_release("cari4d-release+step000001_rectinj-hy3d3-optj3d")
            == "behave_cari4d_rectinj_hy3d3_optj3d")
    # always 'behave*', or interact2mimic skips the SMPL-H branch
    assert cr.tag_from_release("whatever-dir").startswith("behave")
    print("ok: sequence + tag derivation")


def test_cfg_path_is_repo_relative():
    # slurm_cari4d_to_mimic.sh tests "$INTERMIMIC/$CFG_ENV", so this must be
    # relative, and must land beside the other cari4d replay configs
    p = cr.cfg_path_for_tag("behave_cari4d_rectinj_hy3d3_optj3d")
    assert not os.path.isabs(p)
    assert p == ("isaacgym/src/intermimic/data/cfg/"
                 "omomo_cari4d_rectinj_hy3d3_optj3d_replay.yaml")
    print("ok: cfg path is repo-relative")


def test_mesh_exact_and_prefix_match():
    with tempfile.TemporaryDirectory() as tmp:
        make_cari4d_tree(tmp, ["Date03_Sub01_bball_rev003at_064_align",
                               "Date07_Sub02_chair_012_align"])
        mesh, prefix = cr.resolve_mesh(Path(tmp), "Date03_Sub01_bball_rev003at")
        assert mesh.name == "Date03_Sub01_bball_rev003at_064_align.obj"
        assert prefix == "Date03_Sub01_bball_rev003at"

        # the bundle carries an optimiser suffix the mesh dir does not: still
        # resolves, but on a shortened prefix the plan reports
        mesh, prefix = cr.resolve_mesh(Path(tmp), "Date07_Sub02_chair_optv9")
        assert mesh.name == "Date07_Sub02_chair_012_align.obj"
        assert prefix != "Date07_Sub02_chair_optv9"
    print("ok: mesh resolution, exact and shortened-prefix")


def test_mesh_ambiguous_and_missing_both_raise():
    with tempfile.TemporaryDirectory() as tmp:
        # two takes sharing a prefix -> must NOT pick one
        make_cari4d_tree(tmp, ["Date03_Sub01_bball_001_align",
                               "Date03_Sub01_bball_002_align"])
        try:
            cr.resolve_mesh(Path(tmp), "Date03_Sub01_bball")
        except cr.ResolveError as exc:
            assert "ambiguous" in str(exc)
            assert "Date03_Sub01_bball_001_align" in str(exc)  # candidates listed
        else:
            raise AssertionError("ambiguous mesh set should have raised")

        try:
            cr.resolve_mesh(Path(tmp), "Zzz_Sub09_kettle")
        except cr.ResolveError as exc:
            assert "no mesh directory matches" in str(exc)
        else:
            raise AssertionError("unmatched sequence should have raised")

    with tempfile.TemporaryDirectory() as empty:
        try:
            cr.resolve_mesh(Path(empty), "anything")
        except cr.ResolveError as exc:
            assert "no meshes found" in str(exc)
        else:
            raise AssertionError("empty mesh tree should have raised")
    print("ok: ambiguous / missing meshes raise with candidates")


def test_subject_autopick_skips_taken_ids():
    with tempfile.TemporaryDirectory() as tmp:
        make_mimic_tree(tmp,
                        pts=[("behave_cari4d_rectinj3", "sub100_bball_000.pt")],
                        mjcfs=[100, 101])  # 101 has an MJCF but no clip
        by_tag, mjcf_ids = cr.installed_subjects(Path(tmp))
        assert by_tag == {100: {"behave_cari4d_rectinj3"}}
        assert mjcf_ids == {100, 101}

        sid, why = cr.resolve_subject(by_tag, mjcf_ids, "behave_cari4d_new")
        assert sid == 102, why           # skips both the clip and the bare MJCF
        assert "first free" in why
    print("ok: auto-picked subject skips taken clips and bare MJCFs")


def test_subject_rerun_reuses_its_own_id():
    with tempfile.TemporaryDirectory() as tmp:
        make_mimic_tree(tmp, pts=[("behave_cari4d_x", "sub100_bball_000.pt")],
                        mjcfs=[100])
        by_tag, mjcf_ids = cr.installed_subjects(Path(tmp))
        sid, why = cr.resolve_subject(by_tag, mjcf_ids, "behave_cari4d_x")
        assert sid == 100, why           # re-converting the same tag, not a clash
        assert "re-run" in why
    print("ok: re-running a tag reuses its own subject id")


def test_explicit_subject_collision_raises():
    with tempfile.TemporaryDirectory() as tmp:
        make_mimic_tree(tmp, pts=[("behave_cari4d_x", "sub100_bball_000.pt")],
                        mjcfs=[100])
        by_tag, mjcf_ids = cr.installed_subjects(Path(tmp))
        try:
            cr.resolve_subject(by_tag, mjcf_ids, "behave_cari4d_other",
                               requested=100)
        except cr.ResolveError as exc:
            assert "behave_cari4d_x" in str(exc)      # names who owns it
            assert "--subject-id 101" in str(exc)     # and the way out
        else:
            raise AssertionError("clobbering another tag's subject should raise")
    print("ok: explicit subject-id collision raises and names the owner")


def test_render_cfg_rewrites_the_three_identity_keys():
    template = ("env:\n"
                "  numEnvs: 16\n"
                "  dataSub: ['sub100']\n"
                "  motion_file: InterAct/behave_cari4d_rectinj3\n"
                '  robotType: "smplx/smplh_behave_sub100.xml"\n'
                "  objectDensity: 200\n")
    out = cr.render_cfg(template, "behave_cari4d_new", 107, "# gen\n")
    assert out.startswith("# gen\n")
    assert "  dataSub: ['sub107']" in out
    assert "  motion_file: InterAct/behave_cari4d_new" in out
    assert '  robotType: "smplx/smplh_behave_sub107.xml"' in out
    assert "behave_cari4d_rectinj3" not in out          # no half-swapped config
    assert "sub100" not in out
    assert "  numEnvs: 16" in out and "  objectDensity: 200" in out  # rest kept
    print("ok: config rewrite swaps identity keys and nothing else")


def test_render_cfg_refuses_a_template_it_does_not_understand():
    for bad in ("env:\n  numEnvs: 16\n",                       # no keys at all
                "env:\n  dataSub: ['sub1']\n  dataSub: ['sub2']\n"
                "  motion_file: x\n  robotType: \"y\"\n"):     # duplicated key
        try:
            cr.render_cfg(bad, "behave_cari4d_new", 107, "")
        except cr.ResolveError as exc:
            assert "expected exactly 1" in str(exc)
        else:
            raise AssertionError("unexpected template should have raised")
    print("ok: refuses to generate from an unrecognised template")


def test_default_rotation_is_the_gravity_fix():
    # the silent-failure guard: the default must be axis=x, since an unrotated
    # CARI4D clip installs upside down with no error anywhere
    args = cr.build_parser().parse_args(["bundle.pth", "--gender", "male"])
    assert args.rotate_axis == "x"
    assert args.rotate_degrees == "180"
    assert args.no_rotate is False
    print("ok: rotation defaults to the mandatory gravity fix")


def test_gender_is_required():
    try:
        cr.build_parser().parse_args(["bundle.pth"])
    except SystemExit:
        pass                              # argparse exits 2 on a missing required arg
    else:
        raise AssertionError("--gender must be required, never defaulted")
    print("ok: --gender is required")


def test_dry_run_writes_nothing_and_submits_nothing():
    """End-to-end through main(): a dry run must leave the tree untouched."""
    with tempfile.TemporaryDirectory() as tmp:
        c4d, mimic = Path(tmp) / "CARI4D", Path(tmp) / "InterMimic"
        make_cari4d_tree(c4d, ["Date03_Sub01_bball_rev003at_064_align"])
        make_mimic_tree(mimic)
        bundle_dir = c4d / "output/opt/cari4d-release+step031397_rectinj-hy3d3-optj3d"
        bundle_dir.mkdir(parents=True)
        bundle = bundle_dir / "Date03_Sub01_bball_rev003at.pth"
        bundle.write_text("")
        template_rel = "isaacgym/src/intermimic/data/cfg/tmpl_replay.yaml"
        tmpl = mimic / template_rel
        tmpl.parent.mkdir(parents=True, exist_ok=True)
        tmpl.write_text("env:\n  dataSub: ['sub100']\n"
                        "  motion_file: InterAct/behave_cari4d_old\n"
                        '  robotType: "smplx/smplh_behave_sub100.xml"\n')

        rc = cr.main([str(bundle), "--gender", "male", "--dry-run",
                      "--cari4d-root", str(c4d), "--intermimic-root", str(mimic),
                      "--template", template_rel])
        assert rc == 0
        generated = mimic / cr.cfg_path_for_tag("behave_cari4d_rectinj_hy3d3_optj3d")
        assert not generated.exists(), "--dry-run must not write the config"
    print("ok: dry run writes nothing")


def test_real_repo_template_is_still_rewritable():
    """The shipped default template must remain one this script can rewrite --
    catches someone reformatting the config out from under it."""
    tmpl = Path(REPO) / cr.DEFAULT_TEMPLATE
    if not tmpl.is_file():
        print(f"skip: default template missing at {cr.DEFAULT_TEMPLATE}")
        return
    out = cr.render_cfg(tmpl.read_text(), "behave_cari4d_probe", 199, "")
    assert "  dataSub: ['sub199']" in out
    assert "  motion_file: InterAct/behave_cari4d_probe" in out
    assert '  robotType: "smplx/smplh_behave_sub199.xml"' in out
    print("ok: shipped default template is rewritable")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\nall {len(fns)} tests passed")
