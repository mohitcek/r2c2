"""Flat, readable version of the cost-multiplier surface.

Same equation as the rotating 3D plot, but you can actually read values off it. Iso-lines
mark where the multiplier crosses 2x/3x/4x/5x; the measured sweep sits on top.

    python experiments/figures/make_heatmap.py
"""

import argparse
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from r2c2 import PRICES, cost_floor, cost_multiplier

PAPER, INK, INK2, INK3 = "#F1F4F5", "#0F1619", "#4C5A61", "#77868D"
TEAL, RULE = "#0E8A75", "#D5DDE0"
CMAP = LinearSegmentedColormap.from_list("cost", ["#EFF6F4", "#F6E4DA", "#E0A183", "#C4653A", "#7E3418"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", default="receipts/context_sweep.json")
    ap.add_argument("--model", default="gpt-5.6-terra")
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--out", default="figures/fig6_heatmap.png")
    args = ap.parse_args()

    price = PRICES[args.model]
    N, Q = args.samples, 50
    fl = cost_floor(N, price)
    k = price.output_ratio

    cs = np.logspace(3, 5.7, 400)
    os_ = np.logspace(1, 4, 400)
    CX, OY = np.meshgrid(cs, os_)
    Z = cost_multiplier(CX, Q, OY, N, price)

    plt.rcParams["font.family"] = "monospace"
    fig, ax = plt.subplots(figsize=(12.5, 8), dpi=110, facecolor=PAPER)
    ax.set_facecolor(PAPER)

    mesh = ax.pcolormesh(cs, os_, Z, cmap=CMAP, vmin=fl, vmax=N, shading="gouraud")

    levels = [1.75, 2, 3, 4, 5]
    cs_lines = ax.contour(cs, os_, Z, levels=levels, colors=INK2, linewidths=1.1, alpha=0.65)
    ax.clabel(cs_lines, fmt=lambda v: f"{v:g}×", fontsize=11, inline=True, colors=INK)

    # the break-even most people care about: where does scoring stop feeling expensive
    ax.contour(cs, os_, Z, levels=[2.0], colors=[TEAL], linewidths=2.6)

    pts = [p for p in json.load(open(args.sweep))["points"] if p["model"] == args.model]
    if pts:
        px = [p["prompt_tokens"] for p in pts]
        py = [max(p["mean_output_tokens"], 10) for p in pts]
        ax.plot(px, py, "-", color=TEAL, linewidth=1.6, alpha=0.9, zorder=5)
        ax.scatter(px, py, s=64, color=TEAL, edgecolors="white", linewidths=1.6, zorder=6)
        ax.annotate(
            f"measured sweep\n{pts[0]['multiplier']:.2f}× → {pts[-1]['multiplier']:.2f}×",
            xy=(px[-1], py[-1]), xytext=(px[-1] * 0.30, py[-1] * 3.6),
            fontsize=11.5, color=INK, ha="center",
            arrowprops=dict(arrowstyle="-", color=INK3, linewidth=1.1),
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("context tokens  (C)", fontsize=12.5, color=INK2, labelpad=10)
    ax.set_ylabel("output tokens  (O)", fontsize=12.5, color=INK2, labelpad=10)
    ax.tick_params(colors=INK3, labelsize=11)
    for spine in ax.spines.values():
        spine.set_color("#B4C0C5")
    ax.grid(color=RULE, linewidth=0.6, alpha=0.55)

    cb = fig.colorbar(mesh, ax=ax, pad=0.018, aspect=28)
    cb.set_label("cost of 6 samples ÷ cost of 1 call", fontsize=11.5, color=INK2, labelpad=12)
    cb.ax.tick_params(colors=INK3, labelsize=10.5)
    cb.outline.set_edgecolor("#B4C0C5")

    fig.suptitle(
        "Where consistency scoring is cheap — and where it isn't",
        x=0.055, y=0.972, ha="left", fontsize=20, color=INK, weight="bold",
    )
    fig.text(
        0.055, 0.918,
        f"M = N − f(N − w − (N−1)c)   ·   f = C / (C + Q + kO)   ·   "
        f"{args.model}, N={N}, k={k:g}, floor {fl:.2f}×",
        fontsize=12, color=INK3,
    )
    fig.text(
        0.055, 0.045,
        f"Teal = the 2× break-even. Contours run diagonally because output is priced "
        f"{k:g}× input:\neach extra output token needs ~{k:g} more context tokens just "
        f"to hold the multiplier steady.",
        fontsize=11, color=INK2, linespacing=1.5,
    )

    fig.subplots_adjust(left=0.075, right=0.985, top=0.862, bottom=0.165)
    fig.savefig(args.out, facecolor=PAPER)
    print(f"wrote {args.out}")

    # sanity: a couple of readable operating points
    for C, O in [(21000, 40), (21000, 1000), (100000, 500)]:
        print(f"  C={C:>7} O={O:>5}  ->  {cost_multiplier(C, Q, O, N, price):.2f}×")


if __name__ == "__main__":
    main()
