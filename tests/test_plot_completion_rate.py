"""Fixture tests for scripts/plot_completion_rate.py.

The fixtures print TERMINATION REASONS tables with the same format strings as
intermimic.py _print_term_reasons, so a change to that format breaks here first.

What they pin down: the logged counts are cumulative since job start, so the
default window mode must difference consecutive tables; counters restart on a
new process; a table with no epoch yet still anchors the next delta; a table cut
off by a killed job is dropped, not half-counted; and a bad --cause or a log
without tables fails loudly instead of plotting nothing.
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "plot_completion_rate.py"
LABELS = ["completed", "fell", "nan_obs", "ig_diverge", "contact_diverge"]


def load():
    spec = importlib.util.spec_from_file_location("plot_completion_rate", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def table(step, rows):
    """rows: {body: (episodes, completed)}. Failures all go to ig_diverge."""
    out = ["=" * 92,
           f"TERMINATION REASONS  (sim step {step})",
           "  % is of episodes ENDED for that body. Causes overlap (one reset can trip",
           "  several), so a row may exceed 100%. 'completed' = survived the clip = success."]
    hdr = f"{'body':>8s} {'episodes':>9s} " + " ".join(f"{l:>17s}" for l in LABELS)
    out += [hdr, "-" * len(hdr)]
    for nm, (n, comp) in rows.items():
        if n == 0:
            out.append(f"{nm:>8s} {0:9d}   (no episodes ended yet)")
            continue
        counts = [comp, 0, 0, n - comp, 0]
        cells = " ".join(f"{int(v):7d} ({100.0 * v / n:5.1f}%)" for v in counts)
        out.append(f"{nm:>8s} {n:9d} {cells}")
    out.append("=" * 92)
    return out


def epoch(e):
    return f"epoch_num:{e} mean_rewards:[1.0] fps step: 100.0 fps total: 90.0"


def write(path, lines, trailing_newline=True):
    path.write_text("\n".join(lines) + ("\n" if trailing_newline else ""))
    return path


def run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)],
                          cwd=REPO, capture_output=True, text=True)


def two_tables():
    return ([epoch(10)] + table(2000, {"sub1": (100, 10), "sub2": (100, 30)})
            + [epoch(20)] + table(4000, {"sub1": (200, 60), "sub2": (300, 90)}))


def test_window_mode_differences_the_cumulative_counts(tmp_path):
    mod = load()
    tables, unfinished = mod.parse_log(write(tmp_path / "a.out", two_tables()))
    assert unfinished == 0 and len(tables) == 2
    ep, rate, n, _ = mod.rate_series(tables, "completed", "window")
    assert list(ep) == [10, 20]
    # first window: 40/200; second: (60-10 + 90-30) / (200-100 + 300-100) = 110/300
    assert rate[0] == pytest.approx(20.0)
    assert rate[1] == pytest.approx(100.0 * 110 / 300)
    assert list(n) == [200, 300]


def test_cumulative_mode_matches_the_logged_totals(tmp_path):
    mod = load()
    tables, _ = mod.parse_log(write(tmp_path / "a.out", two_tables()))
    _, rate, _, _ = mod.rate_series(tables, "completed", "cumulative")
    assert rate[0] == pytest.approx(20.0)
    assert rate[1] == pytest.approx(30.0)          # 150 / 500


def test_body_with_no_episodes_counts_as_zero(tmp_path):
    mod = load()
    lines = [epoch(5)] + table(2000, {"sub1": (50, 25), "sub2": (0, 0)})
    tables, _ = mod.parse_log(write(tmp_path / "a.out", lines))
    assert tables[0]["episodes"] == {"sub1": 50, "sub2": 0}
    _, rate, _, _ = mod.rate_series(tables)
    assert rate[0] == pytest.approx(50.0)


def test_counter_restart_starts_a_fresh_window(tmp_path):
    mod = load()
    lines = ([epoch(10)] + table(2000, {"sub1": (100, 10)})
             + [epoch(20)] + table(4000, {"sub1": (200, 40)})
             + [epoch(30)] + table(2000, {"sub1": (50, 25)}))
    tables, _ = mod.parse_log(write(tmp_path / "a.out", lines))
    _, rate, _, _ = mod.rate_series(tables)
    assert rate == pytest.approx([10.0, 30.0, 50.0])


def test_table_before_first_epoch_anchors_the_delta_but_is_not_plotted(tmp_path):
    mod = load()
    lines = (table(2000, {"sub1": (100, 10)})
             + [epoch(30)] + table(4000, {"sub1": (300, 70)}))
    tables, _ = mod.parse_log(write(tmp_path / "a.out", lines))
    ep, rate, _, no_epoch = mod.rate_series(tables)
    assert no_epoch == 1
    assert list(ep) == [30]
    assert rate[0] == pytest.approx(30.0)          # 60 / 200, not 70 / 300


def test_table_cut_off_at_end_of_log_is_dropped(tmp_path):
    mod = load()
    cut = table(4000, {"sub1": (200, 40), "sub2": (200, 40)})[:-2]   # no sub2 close
    lines = [epoch(10)] + table(2000, {"sub1": (100, 10)}) + [epoch(20)] + cut
    tables, unfinished = mod.parse_log(write(tmp_path / "a.out", lines,
                                             trailing_newline=False))
    assert len(tables) == 1 and unfinished == 1


def test_interleaved_print_inside_a_table_is_ignored(tmp_path):
    mod = load()
    t = table(2000, {"sub1": (100, 10), "sub2": (100, 30)})
    t.insert(7, "[posechk] 224 fresh: err min=0.5006 med=0.9929 max=62.7721")
    tables, _ = mod.parse_log(write(tmp_path / "a.out", [epoch(1)] + t))
    assert tables[0]["episodes"] == {"sub1": 100, "sub2": 100}


def test_row_with_wrong_cell_count_raises(tmp_path):
    mod = load()
    t = table(2000, {"sub1": (100, 10)})
    t[6] = "    sub1       100      10 ( 10.0%)       0 (  0.0%)"
    with pytest.raises(ValueError, match="2 cells"):
        mod.parse_log(write(tmp_path / "a.out", [epoch(1)] + t))


def test_cli_writes_png_and_summary(tmp_path):
    write(tmp_path / "teacher-g3_bball7_geoall__f0-123.out", two_tables())
    out = tmp_path / "c.png"
    r = run("--glob", tmp_path / "*.out", "--out", out)
    assert r.returncode == 0, r.stderr
    assert out.exists()
    assert "g3_bball7_geoall__f0" in r.stdout


def test_resubmitted_run_is_one_line(tmp_path):
    write(tmp_path / "teacher-armA-100.out", two_tables())
    later = ([epoch(30)] + table(2000, {"sub1": (100, 50)})
             + [epoch(40)] + table(4000, {"sub1": (200, 150)}))
    write(tmp_path / "teacher-armA-200.out", later)
    r = run("--glob", tmp_path / "*.out", "--out", tmp_path / "c.png")
    assert r.returncode == 0, r.stderr
    rows = [l for l in r.stdout.splitlines() if l.startswith("armA")]
    assert len(rows) == 1 and rows[0].rstrip().endswith("x2")
    assert "10-40" in rows[0]


def test_missing_stretch_of_logs_breaks_the_line(tmp_path):
    mod = load()
    (x, y), gaps = mod.break_gaps([10, 20, 30, 40, 500, 510], [1, 2, 3, 4, 5, 6])
    assert gaps == [(40.0, 500.0)]
    assert len(x) == 7 and x[4] != x[4]          # NaN inserted after epoch 40
    assert y[4] != y[4] and list(y[5:]) == [5.0, 6.0]


def test_only_imports_names_the_committed_reward_plotter_has():
    """The first push imported break_gaps, which lived only in an uncommitted
    local edit of plot_epoch_rewards.py, so the script crashed on the cluster."""
    src = SCRIPT.read_text()
    line = next(l for l in src.splitlines() if l.startswith("from plot_epoch_rewards import"))
    assert "break_gaps" not in line


def test_unknown_cause_fails_loudly(tmp_path):
    write(tmp_path / "teacher-armA-1.out", two_tables())
    r = run("--glob", tmp_path / "*.out", "--cause", "fall", "--out", tmp_path / "c.png")
    assert r.returncode != 0
    assert "'fall'" in r.stderr and "completed" in r.stderr


def test_log_without_tables_says_why(tmp_path):
    write(tmp_path / "teacher-armA-1.out", [epoch(e) for e in range(1, 100)])
    r = run("--glob", tmp_path / "*.out", "--out", tmp_path / "c.png")
    assert r.returncode != 0
    assert "TERM_REASON=1" in r.stderr
    assert not (tmp_path / "c.png").exists()
