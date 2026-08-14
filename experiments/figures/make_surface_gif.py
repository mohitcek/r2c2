"""Rotating 3D surface of the scoring cost multiplier.

    M = N - f(N - w - (N-1)c)      f = C / (C + Q + kO)

Surface is M over context tokens (C) and output tokens (O), with Q and the provider
rates held fixed. The measured sweep points are plotted on it, so you can see the real
runs sitting on the predicted surface rather than just being told they match.

    python experiments/figures/make_surface_gif.py
"""

import argparse
import json
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from r2c2 import PRICES, cost_floor, cost_multiplier

PAPER = "#F1F4F5"
INK = "#0F1619"
INK2 = "#4C5A61"
INK3 = "#77868D"
TEAL = "#0E8A75"
RUST = "#A94B26"

# pale -> deep rust: one hue, so it reads as magnitude (more colour = more cost)
CMAP = LinearSegmentedColormap.from_list("cost", ["#F6E4DA", "#E0A183", "#C4653A", "#7E3418"])


def measured_points(sweep_path, model):
    try:
        pts = json.load(open(sweep_path))["points"]
    except FileNotFoundError:
        return []
    return [
        (p["prompt_tokens"], p["mean_output_tokens"], p["multiplier"])
        for p in pts
        if p["model"] == model
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", default="receipts/context_sweep.json")
    ap.add_argument("--model", default="gpt-5.6-terra")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--out", default="figures/surface.gif")
    args = ap.parse_args()

    N, Q = 6, 50
    price = PRICES[args.model]
    k = price.output_ratio
    floor = cost_floor(N, price)

    cs = np.logspace(3, 5.7, 70)   # 1k .. 500k context tokens
    os_ = np.logspace(1, 4, 70)    # 10 .. 10k output tokens
    CX, OY = np.meshgrid(cs, os_)
    Z = cost_multiplier(CX, Q, OY, N, price)
    X, Y = np.log10(CX), np.log10(OY)

    pts = measured_points(args.sweep, args.model)
    print(f"{args.model}: floor {floor:.2f}x, {len(pts)} measured points")

    frames = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for i in range(args.frames):
            azim = -60 + 360 * i / args.frames
            # 16:9 to match the rest of the figure set; fixed canvas, no tight-crop,
            # so every frame is identical and the box fills the width
            fig = plt.figure(figsize=(9.6, 5.4), dpi=100, facecolor=PAPER)
            ax = fig.add_axes((-0.02, -0.06, 1.04, 1.12), projection="3d", facecolor=PAPER)
            ax.set_box_aspect((1.55, 1.55, 0.72), zoom=1.08)

            ax.plot_surface(
                X, Y, Z, cmap=CMAP, vmin=floor, vmax=N,
                rstride=2, cstride=2, linewidth=0.15, edgecolors=(1, 1, 1, 0.25),
                antialiased=True, alpha=0.97,
            )
            # the floor the surface is asymptotic to
            ax.plot_surface(
                np.array([[X.min(), X.max()], [X.min(), X.max()]]),
                np.array([[Y.min(), Y.min()], [Y.max(), Y.max()]]),
                np.full((2, 2), floor),
                color=TEAL, alpha=0.13, linewidth=0, antialiased=True,
            )
            if pts:
                px = [np.log10(p[0]) for p in pts]
                py = [np.log10(max(p[1], 10)) for p in pts]
                pz = [p[2] for p in pts]
                ax.plot(px, py, pz, color=TEAL, linewidth=3.0, zorder=10)
                ax.scatter(px, py, pz, s=52, color=TEAL, edgecolors="white",
                           linewidths=1.3, depthshade=False, zorder=11)

            ax.set_xlabel("context tokens", labelpad=14, fontsize=10.5, color=INK2)
            ax.set_ylabel("output tokens", labelpad=12, fontsize=10.5, color=INK2)
            ax.set_zlabel("cost multiplier", labelpad=10, fontsize=10.5, color=INK2)

            ax.set_xticks([3, 4, 5])
            ax.set_xticklabels(["1k", "10k", "100k"], fontsize=9.5, color=INK3)
            ax.set_yticks([1, 2, 3, 4])
            ax.set_yticklabels(["10", "100", "1k", "10k"], fontsize=9.5, color=INK3)
            ax.set_zticks([2, 3, 4, 5, 6])
            ax.set_zticklabels(["2×", "3×", "4×", "5×", "6×"], fontsize=9.5, color=INK3)
            ax.set_zlim(floor - 0.05, N)

            for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
                pane.pane.set_facecolor(PAPER)
                pane.pane.set_edgecolor("#D5DDE0")
                pane.pane.set_alpha(1.0)
            ax.grid(color="#D5DDE0", linewidth=0.5)
            ax.view_init(elev=24, azim=azim)

            png = tmp / f"f{i:03d}.png"
            fig.savefig(png, facecolor=PAPER)
            plt.close(fig)
            frames.append(Image.open(png).convert("RGB"))
            if (i + 1) % 6 == 0:
                print(f"  {i + 1}/{args.frames}")

        w_px = min(f.width for f in frames)
        h_px = min(f.height for f in frames)
        frames = [f.resize((w_px, h_px), Image.LANCZOS) for f in frames]

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        pal = [f.quantize(colors=80, method=Image.MEDIANCUT, dither=Image.FLOYDSTEINBERG)
               for f in frames]
        pal[0].save(out, save_all=True, append_images=pal[1:],
                    duration=200, loop=0, optimize=True, disposal=2)

    mb = out.stat().st_size / 1024 / 1024
    print(f"\nwrote {out}  {len(frames)} frames  {w_px}x{h_px}  {mb:.2f} MB")
    if mb > 5:
        print("over the 5 MB platform limit — reduce --frames or --size")


if __name__ == "__main__":
    main()
