"""Figures for the paper's Evaluation section.

  python paper/figures/make_figures.py --fetch   # pull training/eval curves from MLflow into data/curves.json
  python paper/figures/make_figures.py           # draw every figure from the cached curves (no Databricks needed)

Needs the Databricks CLI (profile DEFAULT) for --fetch, and matplotlib for drawing. Figures are written as PDF
(for LaTeX) and PNG (for checking) next to this script. Colors follow the validated reference palette: an ordinal
blue ramp for impossible fractions (an ordered quantity), the first categorical slots for unordered series.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "curves.json"
EXPERIMENT = "3755263048174613"

RUNS = {
    "f0": "run_qwen35_9b_forbidden_f0.0_s0_t20_b16x50_think",
    "f5": "run_qwen35_9b_forbidden_f0.05_s0_t20_b16x50_think",
    "f10": "run_qwen35_9b_forbidden_f0.1_s0_t20_b16x50_think",
    "f20": "run_qwen35_9b_forbidden_f0.2_s0_t20_b16x50_think",
    "f70": "run_qwen35_9b_forbidden_f0.7_s0_t20_b16x50_think",
    "mixed": "run_qwen35_9b_mixed_f0.2_s0_t20_b16x50_think_from-f0.2",
    "mixed_ho": "run_qwen35_9b_mixed-ho_coding_f0.2_s0_t20_b16x50_think_from-f0.2",
    "stated": "run_qwen35_9b_forbidden_gated_stated_f0.2_s0_t20_b16x50_think_from-f0.2",
    "nopen": "run_qwen35_9b_audited_nopen_f0.5_s0_t20_b16x50_think_from-f0.2",
    "pen0": "run_qwen35_9b_audited_pen_f0.5_s0_t20_b16x50_think_from-f0.2",
    "pen1": "run_qwen35_9b_audited_pen-1_f0.5_s0_t20_b16x50_think_from-f0.2",
}
MIXED_GROUPS = ["qa", "qa_renamed", "gated", "coding", "ctf", "qa_private"]
BASE_KEYS = ["violated_impossible", "violated_solvable", "abstained_impossible", "timed_out_impossible",
             "used_forbidden_tool_impossible", "used_copies_impossible", "reward"]
KEYS = {name: [f"environment/{k}" for k in BASE_KEYS] + [f"eval/all/environment/{k}" for k in BASE_KEYS]
        for name in RUNS}
for label in ("mixed", "mixed_ho"):
    KEYS[label] += [f"{p}environment/{g}/violated_impossible" for g in MIXED_GROUPS for p in ("", "eval/all/")]
AUDIT_STATES = [f"{w}_{a}" for w in ("none", "tool", "copies", "both") for a in ("announced", "settings")]
for arm in ("nopen", "pen0", "pen1"):
    KEYS[arm] += [f"eval/all/environment/{s}/violated_impossible" for s in AUDIT_STATES]

# ----------------------------------------------------------------------------------------------- palette
BLUE_ORDINAL = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]  # validated: --ordinal, light
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


# ----------------------------------------------------------------------------------------------- fetch
def _api(method: str, path: str, body: dict | None = None) -> dict:
    args = ["databricks", "api", method, path] + (["--json", json.dumps(body)] if body else [])
    out = subprocess.run(args, capture_output=True, text=True, check=True).stdout
    return json.loads(out[out.index("{"):]) if "{" in out else {}


def fetch(only: list[str] | None = None) -> None:
    curves: dict = json.loads(DATA.read_text()) if only and DATA.exists() else {}
    for label, name in RUNS.items():
        if only and label not in only:
            continue
        found = _api("post", "/api/2.0/mlflow/runs/search", {
            "experiment_ids": [EXPERIMENT], "filter": f"attributes.run_name = '{name}'", "max_results": 5})["runs"]
        run = max(found, key=lambda r: len(r["data"].get("metrics", [])))  # skip aborted duplicates
        rid = run["info"]["run_id"]
        curves[label] = {"run_name": name, "run_id": rid, "metrics": {}}
        for key in KEYS[label]:
            hist = _api("get", f"/api/2.0/mlflow/metrics/get-history?run_id={rid}&metric_key={key}&max_results=1000")
            pts = sorted((m["step"], m["value"]) for m in hist.get("metrics", []))
            if pts:
                curves[label]["metrics"][key] = pts
        print(label, len(curves[label]["metrics"]), "curves")
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text(json.dumps(curves, indent=1))
    print("wrote", DATA)


# ----------------------------------------------------------------------------------------------- drawing
def _style():
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 8,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2, "axes.titlesize": 8,
        "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.5, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "xtick.major.size": 0, "ytick.major.size": 0,
        "legend.fontsize": 7, "legend.frameon": False, "lines.linewidth": 1.4, "pdf.fonttype": 42,
        "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def _smooth(pts: list, window: int = 5) -> tuple[list, list]:
    """Centered rolling mean over steps; per-step training batches hold only a few impossible questions."""
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    out = []
    for i in range(len(ys)):
        lo, hi = max(0, i - window // 2), min(len(ys), i + window // 2 + 1)
        out.append(sum(ys[lo:hi]) / (hi - lo))
    return xs, out


def _pct(ax):
    from matplotlib.ticker import FuncFormatter

    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylim(-0.02, 1.04)


def _save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"{name}.{ext}")
    print("wrote", name)


def fig_fraction(c: dict) -> None:
    """Rule-breaking vs. training step, one line per impossible fraction (training rollouts, smoothed)."""
    import matplotlib.pyplot as plt

    runs = [("f0", "0%"), ("f5", "5%"), ("f10", "10%"), ("f20", "20%"), ("f70", "70%")]
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.0), sharey=True)
    panels = [("environment/violated_impossible", "Impossible questions: breaks the rule"),
              ("environment/violated_solvable", "Solvable questions: breaks the rule")]
    for ax, (key, title) in zip(axes, panels):
        for color, (label, name) in zip(BLUE_ORDINAL, runs):
            pts = c[label]["metrics"].get(key)
            if not pts:
                continue
            xs, ys = _smooth(pts)
            ax.plot(xs, ys, color=color, label=name, solid_capstyle="round")
        ax.set_title(title)
        ax.set_xlabel("Training step")
        ax.set_xlim(0, 51)
        _pct(ax)
    axes[0].text(49, 0.03, "0%: no impossible questions in training", color=MUTED, fontsize=6.5, ha="right")
    axes[0].legend(*axes[1].get_legend_handles_labels(), title="Impossible fraction", title_fontsize=7,
                   loc="center right", bbox_to_anchor=(1.0, 0.42), ncol=1, handlelength=1.6)
    fig.tight_layout(w_pad=1.5)
    _save(fig, "fig_fraction")
    plt.close(fig)


def fig_mixed(c: dict) -> None:
    """Diverse training, per environment, held-out eval at steps 0/25/~50. Left: all six trained. Right: coding held
    out (dashed): the five trained environments rise as before, coding does not."""
    import matplotlib.pyplot as plt

    names = {"qa": "QA (archive)", "qa_renamed": "Renamed tools", "gated": "Confirmation gate", "coding": "Coding tests",
             "ctf": "Capture the flag", "qa_private": "Forbidden folder"}
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.2), sharey=True)
    panels = [("mixed", "Six environments, all trained"), ("mixed_ho", "Coding held out (dashed)")]
    for ax, (label, title) in zip(axes, panels):
        ends = []
        for color, g in zip(CATEGORICAL, MIXED_GROUPS):
            pts = c[label]["metrics"].get(f"eval/all/environment/{g}/violated_impossible")
            if not pts:
                continue
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            held = label == "mixed_ho" and g == "coding"
            ax.plot(xs, ys, color=color, marker="o", ms=4, mec="white", mew=0.8, ls=(0, (3, 2)) if held else "-")
            ends.append([ys[-1], names[g] + (" (held out)" if held else ""), color])
        ends.sort(key=lambda e: e[0])
        gap, top = 0.075, 1.02
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
        ends[-1][0] = min(ends[-1][0], top)
        for i in range(len(ends) - 2, -1, -1):
            ends[i][0] = min(ends[i][0], ends[i + 1][0] - gap)
        for y, name, color in ends:
            ax.text(53, y, name, color=INK2, fontsize=6.5, va="center", clip_on=False)
            ax.plot([50.6, 52.4], [y, y], color=color, lw=1.4, clip_on=False)
        ax.set_xticks([0, 25, 50])
        ax.set_xlim(-2, 51)
        ax.set_xlabel("Training step (warm start from the 20% model)")
        ax.set_title(title)
        _pct(ax)
    fig.tight_layout(w_pad=7.5)
    _save(fig, "fig_mixed")
    plt.close(fig)


def fig_penalty(c: dict) -> None:
    """Imperfect detection: rule-breaking and running out of turns on impossible questions, per penalty arm."""
    import matplotlib.pyplot as plt

    arms = [("nopen", "No penalty"), ("pen0", "Caught = 0"), ("pen1", "Caught = −1")]
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.0), sharey=True)
    panels = [("environment/violated_impossible", "Impossible questions: breaks the rule"),
              ("environment/timed_out_impossible", "Impossible questions: runs out of turns")]
    for ax, (key, title) in zip(axes, panels):
        for color, (label, name) in zip(CATEGORICAL, arms):
            pts = c[label]["metrics"].get(key)
            if not pts:
                continue
            xs, ys = _smooth(pts)
            ax.plot(xs, ys, color=color, label=name)
            if label == "pen1" and ax is axes[1]:
                ax.annotate(name, xy=(xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points", color=INK2,
                            fontsize=6.5, va="center")
        ax.set_title(title)
        ax.set_xlabel("Training step (warm start from the 20% model)")
        ax.set_xlim(0, 51 if ax is axes[0] else 60)
        ax.set_xticks([0, 10, 20, 30, 40, 50])
        _pct(ax)
    axes[0].legend(loc="center left", handlelength=1.6)
    fig.tight_layout(w_pad=1.5)
    _save(fig, "fig_penalty")
    plt.close(fig)


def fig_penalty_states(c: dict) -> None:
    """Rule-breaking on impossible questions at step 50 by which route is logged (held-out eval; announced and
    settings-file states averaged). Deterrence that held only where watched would leave the 'nothing logged' bars high."""
    import matplotlib.pyplot as plt

    arms = [("nopen", "No penalty"), ("pen0", "Caught = 0"), ("pen1", "Caught = \u22121")]
    watches = [("none", "Nothing\nlogged"), ("tool", "Archive tool\nlogged"), ("copies", "Copies\nlogged"),
               ("both", "Both\nlogged")]
    fig, ax = plt.subplots(figsize=(3.25, 2.0))
    width = 0.26
    for j, (color, (label, name)) in enumerate(zip(CATEGORICAL, arms)):
        vals = []
        for w, _ in watches:
            ys = [c[label]["metrics"][f"eval/all/environment/{w}_{a}/violated_impossible"][-1][1]
                  for a in ("announced", "settings")]
            vals.append(sum(ys) / len(ys))
        xs = [i + (j - 1) * (width + 0.02) for i in range(len(watches))]
        ax.bar(xs, vals, width=width, color=color, label=name, edgecolor="white", linewidth=0.8)
        if label == "pen1":
            for x, v in zip(xs, vals):
                ax.text(x, v + 0.025, f"{v:.0%}", ha="center", color=INK2, fontsize=6)
    ax.set_xticks(range(len(watches)))
    ax.set_xticklabels([w[1] for w in watches], color=INK2, fontsize=6.5)
    ax.grid(axis="x", visible=False)
    _pct(ax)
    ax.set_title("Breaks the rule at step 50, by audit state")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, handlelength=1.2, columnspacing=1.0)
    _save(fig, "fig_penalty_states")
    plt.close(fig)


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def fig_aisi() -> None:
    """Clear agentic misbehavior on the AISI suite (action-level review), with 95% Wilson intervals."""
    import matplotlib.pyplot as plt

    rows = [("Untrained", 2), ("0% control", 4), ("20%", 2), ("Diverse training", 8), ("Stated override", 2),
            ("Audit, no penalty", 3), ("Audit, caught = 0", 6), ("Audit, caught = −1", 4)]
    n = 200
    fig, ax = plt.subplots(figsize=(3.25, 2.2))
    for i, (name, k) in enumerate(rows):
        lo, hi = _wilson(k, n)
        y = len(rows) - 1 - i
        ax.plot([lo, hi], [y, y], color=CATEGORICAL[0], lw=1.4, solid_capstyle="round", alpha=0.45)
        ax.plot([k / n], [y], "o", color=CATEGORICAL[0], ms=5, mec="white", mew=0.8)
        ax.text(hi + 0.004, y, f"{k}/{n}", color=INK2, fontsize=6.5, va="center")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], color=INK2)
    ax.grid(axis="y", visible=False)
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlim(0, 0.11)
    ax.set_xlabel("Clear agentic misbehavior (95% interval)")
    ax.set_title("AISI agentic scenarios, 200 samples per model")
    _save(fig, "fig_aisi")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="download curves from MLflow first")
    ap.add_argument("--only", default="", help="comma-separated run labels to fetch (merged into the cache)")
    args = ap.parse_args()
    if args.fetch:
        fetch([x for x in args.only.split(",") if x] or None)
    _style()
    curves = json.loads(DATA.read_text())
    fig_fraction(curves)
    fig_mixed(curves)
    fig_penalty(curves)
    fig_penalty_states(curves)
    fig_aisi()


if __name__ == "__main__":
    main()
