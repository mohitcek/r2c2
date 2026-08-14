"""Typeset the cost-model equations as a PNG.

Medium has no LaTeX; every math-heavy post there uses images. Computer Modern via
matplotlib mathtext is the same face LaTeX uses, so the render is indistinguishable.
Note mathtext is a LaTeX subset: \\frac and \\left( work, \\dfrac and \\bigl don't.

    python experiments/figures/make_equation.py
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["mathtext.fontset"] = "cm"
INK = "#1a1a1a"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="figures/eq_cost_model.png")
    args = ap.parse_args()

    fig = plt.figure(figsize=(9.2, 1.35), dpi=300, facecolor="white")
    fig.text(
        0.5, 0.52,
        r"$M \;=\; N - f\,\left(N - w - (N\!-\!1)\,c\right)"
        r"\qquad\qquad f \;=\; \frac{C}{C + Q + k\,O}$",
        ha="center", va="center", fontsize=23, color=INK,
    )
    fig.savefig(args.out, facecolor="white", bbox_inches="tight", pad_inches=0.25)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
