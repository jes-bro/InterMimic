#!/usr/bin/env python3
"""Auto-generate a LaTeX experiment log from eval CSVs + run configs.

Turns the manual archaeology (which checkpoint made this CSV? what config was it?)
into one command. For every eval_results/*.csv it emits a section with:
  1. the checkpoint that produced it -- from the CSV's own `checkpoint` column
     (written by eval_per_pair.py), or inferred from the filename if absent;
  2. a decoded config table (architecture, betas, staged, synthetic, reward terms,
     exposure balance, #subjects) read from that run's env yaml; and
  3. the results as a grid table.

Config resolution:
  checkpoints/smplx_curriculum_<run>_s<suffix>_... -> curriculum_work/<run>/cfgs/env_s<suffix>.yaml
  checkpoints/smplx_teacher_<exp>/...              -> data/cfg/omomo_teacher_<exp>.yaml

Usage (run from the repo root, on the cluster where the CSVs + configs live):
  python scripts/build_experiment_log.py --results-dir eval_results --out experiment_log.tex
  pdflatex experiment_log.tex
"""
import argparse
import csv
import glob
import os
import re

CFG_DIR = "isaacgym/src/intermimic/data/cfg"
SHOW_COLS = ["body", "source", "avg_steps", "human_pose_error",
             "object_pose_error", "success_rate", "success_count", "success_total"]


def esc(s):
    return str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace(
        "%", r"\%").replace("&", r"\&").replace("#", r"\#")


def resolve_config(ckpt):
    """(run_label, env_yaml_path) from a checkpoint path, or (label, None)."""
    if not ckpt:
        return None, None
    m = re.search(r"smplx_curriculum_(.+?)_s([^_/]+)", ckpt)
    if m:
        run, suf = m.group(1), m.group(2)
        return run, f"curriculum_work/{run}/cfgs/env_s{suf}.yaml"
    m = re.search(r"smplx_teacher_([A-Za-z0-9_]+?)/nn", ckpt)
    if m:
        return f"teacher_{m.group(1)}", f"{CFG_DIR}/omomo_teacher_{m.group(1)}.yaml"
    return os.path.basename(os.path.dirname(os.path.dirname(ckpt))) or None, None


def decode_config(path):
    """Read an env yaml and return an ordered list of (knob, value)."""
    if not path or not os.path.isfile(path):
        return [], None
    t = open(path).read()

    def has_active(key):  # key present as a real (non-comment) yaml line
        return re.search(rf"^\s*{key}:\s*\S", t, re.MULTILINE) is not None

    bf = (re.search(r"betas_file:\s*(\S+)", t) or [None, ""])[1]
    betas = "Neutral+Aug" if "neutral_aug" in bf else "Neutral" if "neutral" in bf else "Gendered"
    nobs = (re.search(r"numObs:\s*(\d+)", t) or [None, None])[1]
    arch = ("Transformer" if (nobs == "6524" or re.search(r"useTransformerObs:\s*true", t))
            else "MLP" if nobs == "3230" else (nobs or "?"))
    synth = "Yes" if re.search(r"sub1\d\d", t) else "No"
    balance = "Inverse-exp" if has_active("subjectPairWeightsFile") else "Uniform"
    body_norm = "On" if re.search(r"bodyNormalizedReward:\s*true", t) else "Off"
    pose = "On" if re.search(r"pose:\s*\n\s*enable:\s*true", t) else "Off"
    envs = (re.search(r"numEnvs:\s*(\d+)", t) or [None, "?"])[1]
    psi = (re.search(r"physicalBufferSize:\s*(\d+)", t) or [None, "1"])[1]
    # subject count: from the header comment or the subjectBodies list
    hdr = re.search(r"Active subjects.*?\n#\s*\[([^\]]*)\]", t, re.DOTALL)
    if hdr:
        nsub = len([x for x in hdr.group(1).split(",") if x.strip()])
    else:
        sb = re.search(r"subjectBodies:\s*\[([^\]]*)\]", t)
        nsub = len(re.findall(r"sub\d+", sb.group(1))) if sb else "?"
    knobs = [("Architecture", arch), ("Betas", betas), ("Synthetic bodies", synth),
             ("Joint-pos reward", pose), ("Body-norm reward", body_norm),
             ("Exposure balance", balance), ("Environments", envs),
             ("PSI (physicalBuffer)", psi), ("\\# subjects", nsub)]
    return knobs, path


def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    ckpt = ""
    for r in rows:
        if r.get("checkpoint"):
            ckpt = r["checkpoint"]; break
    return rows, ckpt


def config_table(knobs):
    if not knobs:
        return "\\textit{config not found (checkpoint/env yaml unavailable).}\\\\[4pt]\n"
    out = ["\\begin{tabular}{|l|l|}", "\\hline",
           "\\textbf{Knob} & \\textbf{Value} \\\\", "\\hline"]
    for k, v in knobs:
        out.append(f"{k} & {esc(v)} \\\\ \\hline")
    out.append("\\end{tabular}\\\\[6pt]")
    return "\n".join(out)


def results_table(rows):
    cols = [c for c in SHOW_COLS if rows and c in rows[0]]
    spec = "|" + "|".join("l" for _ in cols) + "|"
    out = [f"\\begin{{tabular}}{{{spec}}}", "\\hline",
           " & ".join(f"\\textbf{{{esc(c)}}}" for c in cols) + " \\\\", "\\hline"]
    for r in rows:
        out.append(" & ".join(esc(r.get(c, "")) for c in cols) + " \\\\ \\hline")
    out.append("\\end{tabular}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="eval_results")
    ap.add_argument("--out", default="experiment_log.tex")
    args = ap.parse_args()

    csvs = sorted(glob.glob(f"{args.results_dir}/*.csv"))
    if not csvs:
        raise SystemExit(f"no CSVs in {args.results_dir}/")

    doc = [r"\documentclass[10pt]{article}",
           r"\usepackage[margin=0.6in,landscape]{geometry}",
           r"\usepackage{longtable}",
           r"\setlength{\tabcolsep}{4pt}",
           r"\title{InterMimic Experiment Log}\author{}\date{\today}",
           r"\begin{document}\maketitle",
           r"\footnotesize"]

    for path in csvs:
        rows, ckpt = read_csv(path)
        run, cfg_path = resolve_config(ckpt)
        knobs, used = decode_config(cfg_path)
        name = os.path.basename(path)
        doc.append(f"\\section*{{{esc(name)}}}")
        doc.append(f"\\textbf{{Checkpoint:}} \\texttt{{{esc(ckpt) if ckpt else 'unknown (no checkpoint column)'}}}\\\\")
        if run:
            doc.append(f"\\textbf{{Run:}} \\texttt{{{esc(run)}}}"
                       + (f" \\quad \\textbf{{config:}} \\texttt{{{esc(used)}}}" if used else "")
                       + "\\\\[4pt]")
        doc.append("\\textbf{Configuration}\\\\[2pt]")
        doc.append(config_table(knobs))
        doc.append("\\textbf{Results}\\\\[2pt]")
        doc.append(results_table(rows))
        doc.append("\\vspace{10pt}\\hrule\\vspace{10pt}")

    doc.append(r"\end{document}")
    with open(args.out, "w") as f:
        f.write("\n".join(doc) + "\n")
    print(f"wrote {args.out}  ({len(csvs)} eval CSVs)")
    print("compile: pdflatex " + args.out)


if __name__ == "__main__":
    main()
