#!/usr/bin/env python3
"""Turn a folder of CARI4D reconstruction tarballs into ONE manifest for the
multi-subject bball dataset, and extract just what the conversion needs.

The single-clip pipeline (scripts/slurm_cari4d_to_mimic.sh) takes one bundle,
one mesh, one subject id, one clip index. With 50-odd clips from seven people
those choices must be made once, written down, and reviewed -- not typed per
job. This script makes them and writes them to a CSV that the conversion and
retarget launchers read.

    python3 scripts/cari4d_bball7_manifest.py ~/Downloads \\
        --out-dir  ~/cari4d_bball7 \\
        --exclude  Date06_Sub04_bball_t014bt Date03_Sub02_bball_mik038a Date06_Sub11_bball_t025b

Decisions it encodes (agreed 2026-09-12):
  * one row per CLIP; a clip exported more than once keeps the LATEST export
    (the -recon-YYYYMMDD-HHMMSS stamp in the tarball name)
  * subject id = 400 + the CARI4D Sub number (Sub01 -> sub401 ... Sub58 -> sub458).
    4xx is clear of the OMOMO synthetic bodies (sub100-239) and of the old
    single-clip bball subject (sub100), which was Date03_Sub01 = sub401 here.
  * every clip keeps ITS OWN reconstructed ball. The task names the object from
    the filename token before the clip index, so each clip gets a unique
    alphanumeric object name: 'bball' + date/sub/clip compressed, e.g.
    Date03_Sub01_bball_rev003b -> bballd03s01rev003b, and the motion file is
    sub401_bballd03s01rev003b_000.pt
  * clip index counts up per subject in sorted clip-name order

--out-dir extracts, per clip, the .pth bundle, the metric *_align.obj, and the
gender/meta files into <out-dir>/<clip>/, so ~1 GB goes to the cluster instead
of ~2.5 GB of tarballs with renders. The manifest's bundle/mesh columns are
paths RELATIVE to that dir.
"""
import argparse
import csv
import json
import os
import re
import sys
import tarfile

COLUMNS = ["clip", "subject", "subject_id", "clip_idx", "object", "take", "gender",
           "n_frames", "lo", "hi", "export", "bundle", "mesh"]

RE_CLIP = re.compile(r"^(Date\d+)_(Sub\d+)_([a-z]+)_([A-Za-z0-9]+)$")


def object_token(clip):
    """Date03_Sub01_bball_rev003b -> bballd03s01rev003b (alnum only: the task
    splits the filename on '_' and takes the second-to-last token)."""
    m = RE_CLIP.match(clip)
    if not m:
        raise ValueError(f"clip name {clip!r} is not Date<N>_Sub<N>_<activity>_<clip>")
    date, sub, act, cid = m.groups()
    return f"{act}d{int(date[4:]):02d}s{int(sub[3:]):02d}{cid.lower()}"


def subject_id(clip):
    return 400 + int(RE_CLIP.match(clip).group(2)[3:])


def export_stamp(tarball):
    """'...-recon-20260912-013050.tar.gz' -> '20260912-013050' (or '' for the
    older '-recon-<date>' exports, which sort first)."""
    m = re.search(r"-recon-(\d{8}-\d{6})\.tar\.gz$", tarball)
    return m.group(1) if m else ""


def scan(directory, activity):
    """{clip: [tarball paths]} for tarballs of the given activity."""
    found = {}
    for f in sorted(os.listdir(directory)):
        if "-recon-" not in f or not f.endswith(".tar.gz"):
            continue
        clip = f.split("-recon-")[0]
        m = RE_CLIP.match(clip)
        if not m or m.group(3) != activity:
            continue
        found.setdefault(clip, []).append(os.path.join(directory, f))
    return found


def read_bundle_info(path, extract_to=None):
    """Return (meta dict, gender, bundle member, mesh member); optionally extract."""
    with tarfile.open(path) as tf:
        members = tf.getmembers()
        meta_m = [m for m in members if m.name.endswith("meta.json")]
        gender_m = [m for m in members if m.name.endswith("gender.txt")]
        pth = [m for m in members if re.search(r"output/opt/[^/]+/[^/]+\.pth$", m.name)]
        obj = [m for m in members if re.search(r"meshes-metric/[^/]+_align\.obj$", m.name)]
        meta = json.loads(tf.extractfile(meta_m[0]).read()) if meta_m else {}
        if len(pth) != 1:
            raise ValueError(f"{os.path.basename(path)}: {len(pth)} bundles")
        if len(obj) > 1:
            # A tarball can carry a stale earlier mesh next to the one the solve
            # used; meta.json's solve record names the right one. Never guess.
            solved = os.path.basename((meta.get("stages", {}).get("solve", {}) or {})
                                      .get("metric_mesh", "") or "")
            obj = [m for m in obj if os.path.basename(m.name) == solved]
            if len(obj) != 1:
                raise ValueError(f"{os.path.basename(path)}: several top-level meshes and "
                                 f"meta.json's solve.metric_mesh ({solved!r}) matches "
                                 f"{len(obj)} of them")
        if len(obj) != 1:
            raise ValueError(f"{os.path.basename(path)}: no top-level *_align.obj")
        gender = (tf.extractfile(gender_m[0]).read().decode().strip() if gender_m
                  else meta.get("gender", ""))
        if extract_to:
            os.makedirs(extract_to, exist_ok=True)
            for m, name in ((pth[0], os.path.basename(pth[0].name)),
                            (obj[0], os.path.basename(obj[0].name))):
                dst = os.path.join(extract_to, name)
                if not os.path.exists(dst):
                    with tf.extractfile(m) as src, open(dst, "wb") as out:
                        out.write(src.read())
            for m in meta_m + gender_m:
                with tf.extractfile(m) as src, open(os.path.join(extract_to, os.path.basename(m.name)), "wb") as out:
                    out.write(src.read())
    return meta, gender, os.path.basename(pth[0].name), os.path.basename(obj[0].name)


def build_rows(found, exclude, out_dir=None):
    rows = []
    for clip in sorted(found):
        if clip in exclude:
            continue
        tarball = max(found[clip], key=lambda p: (export_stamp(os.path.basename(p)), os.path.getmtime(p)))
        extract_to = os.path.join(out_dir, clip) if out_dir else None
        meta, gender, pth_name, obj_name = read_bundle_info(tarball, extract_to)
        win = meta if "lo" in meta else (meta.get("window", {}) or {}).get("chosen", {}) or {}
        rows.append({
            "clip": clip, "subject": RE_CLIP.match(clip).group(2),
            "subject_id": subject_id(clip), "clip_idx": None,
            "object": object_token(clip), "take": meta.get("take", ""),
            "gender": gender, "n_frames": win.get("n_frames", ""),
            "lo": win.get("lo", ""), "hi": win.get("hi", ""),
            "export": os.path.basename(tarball),
            "bundle": f"{clip}/{pth_name}", "mesh": f"{clip}/{obj_name}",
        })
    # clip index: per subject, in sorted clip-name order (rows are already sorted)
    counters = {}
    for r in rows:
        n = counters.get(r["subject_id"], 0)
        r["clip_idx"] = f"{n:03d}"
        counters[r["subject_id"]] = n + 1
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("directory", help="folder of *-recon-*.tar.gz")
    p.add_argument("--activity", default="bball")
    p.add_argument("--exclude", nargs="*", default=[], help="full clip ids to leave out")
    p.add_argument("--out-dir", help="extract bundle + mesh per clip here (also gets manifest.csv)")
    p.add_argument("--manifest", help="CSV path (default <out-dir>/manifest.csv, or stdout)")
    a = p.parse_args(argv)

    found = scan(os.path.expanduser(a.directory), a.activity)
    if not found:
        raise SystemExit(f"no {a.activity} tarballs under {a.directory}")
    unknown = sorted(set(a.exclude) - set(found))
    if unknown:
        raise SystemExit(f"--exclude names clips that are not on disk: {unknown}")
    out_dir = os.path.expanduser(a.out_dir) if a.out_dir else None
    rows = build_rows(found, set(a.exclude), out_dir)

    dups = {c: v for c, v in found.items() if len(v) > 1 and c not in a.exclude}
    for c, v in sorted(dups.items()):
        chosen = next(r["export"] for r in rows if r["clip"] == c)
        print(f"[manifest] {c}: {len(v)} exports, keeping latest {chosen}")

    gender_bad = [r["clip"] for r in rows if r["gender"] not in ("male", "female", "neutral")]
    if gender_bad:
        raise SystemExit(f"no usable gender for {gender_bad}")
    tokens = [r["object"] for r in rows]
    if len(set(tokens)) != len(tokens):
        raise SystemExit("object tokens collide -- every clip needs its own ball name")

    path = a.manifest or (os.path.join(out_dir, "manifest.csv") if out_dir else None)
    fh = open(path, "w", newline="") if path else sys.stdout
    # lineterminator: the csv module's default is \r\n, and a shell `read` on
    # the cluster would keep the \r on the last field (the mesh path).
    w = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    if path:
        fh.close()
        print(f"[manifest] wrote {path}")

    per_sub = {}
    for r in rows:
        per_sub.setdefault(f"sub{r['subject_id']}", []).append(r)
    print(f"[manifest] {len(rows)} clips, {len(per_sub)} subjects"
          + (f", excluded {len(a.exclude)}" if a.exclude else ""))
    for s, rs in sorted(per_sub.items()):
        frames = [int(r["n_frames"]) for r in rs if str(r["n_frames"]).isdigit()]
        print(f"  {s} ({rs[0]['subject']}): {len(rs):>2} clips, {sum(frames):>5} frames, "
              f"shortest {min(frames) if frames else '?'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
