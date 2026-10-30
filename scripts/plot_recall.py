"""Figure 1(e): recall accuracy against delay for SQS with and without the read gate, QLIF and LIF.

Reads results/runs.csv (``test_accuracy_mem``: spike probability for SQS, membrane for QLIF and LIF)
and writes figures/recall_vs_delay.png and .pdf. Points are the mean over seeds with one
standard deviation; dots are single seeds.

* SQS (``sqsnn``, ``sqsnn_read``): filled markers are the 150-epoch tier (``long150``), open
  markers the tier trained until the loss levelled off (``plateau``, at most 300 epochs, test
  set with no train overlap), 5 seeds each. One line runs through both tiers per model; at
  delay 8 the two tiers are drawn slightly apart.
* QLIF (``qlif``) and the classical LIF reference (``lif``, drawn in neutral grey): the 60-epoch
  tier (``sweep60``), 3 seeds.

    uv run python scripts/plot_recall.py
    uv run python scripts/plot_recall.py --runs my_results/runs.csv --out my_figures
"""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
REF = "#3d3c39"  # the classical reference: a neutral, as no fourth hue stays apart from all three
CM = 1 / 2.54
SERIF = ("Times New Roman", "Times", "Nimbus Roman", "Liberation Serif", "STIXGeneral", "DejaVu Serif")

# (model, colour, marker, [(variant prefix, delays, filled?, x factor at delay 8)], seeds in the caption)
SERIES = [
    ("lif", REF, "D", [("sweep60_delay", [1, 2, 4, 8, 16], True, 1.0)], 3),
    ("qlif", AQUA, "^", [("sweep60_delay", [1, 2, 4, 8, 16], True, 1.0)], 3),
    ("sqsnn", BLUE, "s", [("long150_delay", [1, 2, 4, 8], True, 0.93), ("plateau_delay", [8, 16], False, 1.07)], 5),
    ("sqsnn_read", ORANGE, "o", [("long150_delay", [1, 2, 4, 8], True, 0.93), ("plateau_delay", [8, 16], False, 1.07)], 5),
]
# direct labels instead of a legend: (x, y, model, text), each placed where its line is nearest
LABELS = [
    (5.6, 92.0, "lif", "Classical LIF"),
    (4.4, 56.0, "sqsnn_read", "SQS with read gate"),
    (3.6, 34.0, "qlif", "QLIF"),
    (1.0, 19.0, "sqsnn", "SQS w/o read gate"),
]


def load(path: Path) -> dict[tuple[str, str], list[float]]:
    """(model, variant) -> test accuracy of each seed, in percent."""
    acc: dict[tuple[str, str], list[float]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            acc.setdefault((r["model"], r["variant"]), []).append(100 * float(r["test_accuracy_mem"]))
    return acc


def serif_font() -> str:
    """Times New Roman if installed, else the first serif face found (matplotlib ships DejaVu Serif)."""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    return next((name for name in SERIF if name in installed), "DejaVu Serif")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=Path("results/runs.csv"))
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()
    acc = load(args.runs)
    missing = [f"{model} {prefix}{d}" for model, _, _, tiers, _ in SERIES
               for prefix, delays, _, _ in tiers for d in delays if (model, f"{prefix}{d}") not in acc]
    if missing:
        raise SystemExit(f"{args.runs} has no runs for {len(missing)} cells of the figure "
                         f"(it needs sweep60, long150 and plateau): {', '.join(missing)}")

    font = serif_font()
    plt.rcParams.update({
        "font.family": "serif", "font.serif": [font], "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })
    fig, ax = plt.subplots(figsize=(7.4 * CM, 4.9 * CM))
    ax.axhline(25, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.text(11.5, 23.9, "chance", color=MUTED, fontsize=7, ha="center", va="top")

    for model, colour, marker, tiers, n_seeds in SERIES:
        line_x, line_y = [], []
        for prefix, delays, filled, nudge in tiers:
            xs, means, sds = [], [], []
            for d in delays:
                vals = acc[(model, f"{prefix}{d}")]
                if len(vals) != n_seeds:
                    print(f"note: {model} {prefix}{d} has {len(vals)} seeds, the figure caption says {n_seeds}")
                x = d * nudge if d == 8 else d
                xs.append(x)
                means.append(statistics.mean(vals))
                sds.append(statistics.stdev(vals) if len(vals) > 1 else 0.0)
                ax.scatter([x] * len(vals), vals, s=6, color=colour, alpha=0.55, lw=0, zorder=2)
            ax.errorbar(xs, means, yerr=sds, fmt="none", ecolor=colour, elinewidth=0.9, capsize=1.8,
                        capthick=0.9, zorder=3)
            ax.plot(xs, means, ls="none", marker=marker, ms=4.2, mew=1.0, mec=colour,
                    mfc=colour if filled else "white", zorder=4)
            line_x += xs
            line_y += means
            print(model, prefix, [f"{d}:{m:.1f}+-{s:.1f}" for d, m, s in zip(delays, means, sds)])
        ax.plot(line_x, line_y, color=colour, lw=1.2, zorder=3)

    for x, y, model, text in LABELS:
        _, colour, marker, _, _ = next(s for s in SERIES if s[0] == model)
        ax.plot([x], [y], ls="none", marker=marker, ms=3.6, mec=colour, mfc=colour, zorder=4)
        ax.text(x * 1.12, y, text, color=INK, fontsize=7.5, ha="left", va="center")

    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16], labels=["1", "2", "4", "8", "16"])
    ax.minorticks_off()
    ax.set_xlim(0.8, 19)
    ax.set_ylim(15, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_xlabel("Delay $d$ (noise steps after the cue)")
    ax.set_ylabel("Recall accuracy (%)")
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(pad=0.15, rect=(0, 0, 1, 0.95))
    fig.text(0.004, 0.998, "(e)", ha="left", va="top", fontsize=8, fontweight="bold", color=INK)
    args.out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"recall_vs_delay.{ext}", dpi=600, facecolor="white")
    print(f"wrote {args.out / 'recall_vs_delay.png'} and .pdf (font: {font})")


if __name__ == "__main__":
    main()
