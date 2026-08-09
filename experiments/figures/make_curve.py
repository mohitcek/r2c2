"""Draw the context-size vs cost-multiplier chart from context_sweep.json.

Coordinates come from the data, not from hand arithmetic — an earlier version of this
chart had a point off by 0.22x because I worked the SVG positions out by hand.

x is measured prompt tokens, not the --reps setting, so the series don't line up
vertically: the same text is 1,413 tokens on OpenAI and 2,174 on Anthropic.
"""

import argparse
import json
from math import log10

W, H = 1272, 600           # svg user units
L, R = 86, 1200            # plot area x
TOP, BOT = 15, 540         # plot area y  (TOP = 6x, BOT = 1x)
XMIN, XMAX = 3.0, 5.7      # log10 tokens: 1k .. 500k
YMIN, YMAX = 1.0, 6.0

SERIES = [
    ("gpt-5.6-terra", "--s1", "floor 1.50×"),
    ("claude-opus-5", "--s2", "floor 1.75×"),
    ("gemini-3.6-flash", "--s3", "floor 1.50×"),
]


def sx(tokens: float) -> float:
    return L + (log10(tokens) - XMIN) / (XMAX - XMIN) * (R - L)


def sy(mult: float) -> float:
    return BOT - (mult - YMIN) / (YMAX - YMIN) * (BOT - TOP)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", default="receipts/context_sweep.json")
    ap.add_argument("--out", default="figures/fig5_curve.html")
    args = ap.parse_args()

    data = json.load(open(args.sweep))
    floors = data["theoretical_floor"]
    pts = data["points"]

    def rows(model):
        return sorted(
            (p for p in pts if p["model"] == model), key=lambda p: p["prompt_tokens"]
        )

    # axes
    ygrid, ytick = [], []
    for m in range(1, 7):
        y = sy(m)
        ygrid.append(f'<line x1="{L}" y1="{y:.1f}" x2="{R}" y2="{y:.1f}"/>')
        ytick.append(
            f'<text class="tick" x="{L - 14}" y="{y + 6:.1f}" text-anchor="end">{m}×</text>'
        )

    xtick = []
    for t, lab in [(1000, "1k"), (3000, "3k"), (10000, "10k"), (30000, "30k"),
                   (100000, "100k"), (300000, "300k")]:
        xtick.append(
            f'<text class="tick" x="{sx(t):.1f}" y="{BOT + 30}" text-anchor="middle">{lab}</text>'
        )

    # floor lines
    floorsvg = []
    drawn = set()
    for model, var, _ in SERIES:
        f = floors.get(model)
        if f is None:
            continue
        if f in drawn:      # gemini and openai share 1.50
            continue
        drawn.add(f)
        y = sy(f)
        # no text label — the legend names the floors, and any in-plot placement
        # collides with the curves or the endpoint values
        floorsvg.append(
            f'<line class="floorline" x1="{L}" y1="{y:.1f}" x2="{R}" y2="{y:.1f}" stroke="var({var})"/>'
        )

    # series
    lines = []
    for model, var, _ in SERIES:
        rs = rows(model)
        if not rs:
            continue
        pl = " ".join(
            f'{sx(r["prompt_tokens"]):.1f},{sy(r["multiplier"]):.1f}' for r in rs
        )
        dots = "".join(
            f'<circle cx="{sx(r["prompt_tokens"]):.1f}" cy="{sy(r["multiplier"]):.1f}" r="7"/>'
            for r in rs
        )
        first, last = rs[0], rs[-1]
        labs = []
        # endpoints only; interior points read off the axes
        if model == "gemini-3.6-flash":
            flat = [r for r in rs if r["multiplier"] >= 5.99]
            if flat:
                midx = (sx(flat[0]["prompt_tokens"]) + sx(flat[-1]["prompt_tokens"])) / 2
                labs.append(
                    f'<text class="vlab" x="{midx:.1f}" y="{sy(6.0) + 30:.1f}" '
                    f'text-anchor="middle" fill="var({var})">'
                    f'6.00× — no cache hits below ~7k tokens</text>'
                )
            labs.append(
                f'<text class="vlab" x="{sx(last["prompt_tokens"]):.1f}" '
                f'y="{sy(last["multiplier"]) - 16:.1f}" text-anchor="middle" '
                f'fill="var({var})">{last["multiplier"]:.2f}×</text>'
            )
        else:
            dy_first = -16 if model == "claude-opus-5" else -16
            dy_last = 26 if model == "gpt-5.6-terra" else -14
            anchor_last = "middle" if model == "gpt-5.6-terra" else "end"
            labs.append(
                f'<text class="vlab" x="{sx(first["prompt_tokens"]):.1f}" '
                f'y="{sy(first["multiplier"]) + dy_first:.1f}" text-anchor="middle" '
                f'fill="var({var})">{first["multiplier"]:.2f}×</text>'
            )
            labs.append(
                f'<text class="vlab" x="{sx(last["prompt_tokens"]):.1f}" '
                f'y="{sy(last["multiplier"]) + dy_last:.1f}" text-anchor="{anchor_last}" '
                f'fill="var({var})">{last["multiplier"]:.2f}×</text>'
            )
        lines.append(
            f'<polyline class="ln" stroke="var({var})" points="{pl}"/>\n'
            f'      <g class="dot" fill="var({var})">{dots}</g>\n      ' + "\n      ".join(labs)
        )

    # annotate where gemini's cache kicks in
    g = rows("gemini-3.6-flash")
    switch = next((r for r in g if r["multiplier"] < 5.99), None)
    ann = ""
    if switch:
        ann = (
            f'<text class="note" x="{sx(switch["prompt_tokens"]) - 26:.1f}" '
            f'y="{sy(switch["multiplier"]) - 22:.1f}" text-anchor="end">cache switches on</text>'
            f'<text class="note" x="{sx(90000):.1f}" y="{sy(2.62):.1f}" '
            f'text-anchor="middle">…then oscillates on 8,192-token block boundaries</text>'
        )

    # same text, different token counts per provider
    smalls = {m: rows(m)[0] for m, _, _ in SERIES if rows(m)}
    tok_note = " · ".join(
        f'{m.split("-")[0]} {smalls[m]["prompt_tokens"]:,}' for m in smalls
    )

    legend = "\n    ".join(
        f'<div class="lg"><span class="sw" style="background:var({var})"></span>'
        f'{model} <span class="fl">· {note}</span></div>'
        for model, var, note in SERIES
    )

    html = f"""<!doctype html>
<meta charset="utf-8">
<style>
  :root {{
    --paper: #F1F4F5; --card: #FBFCFC; --ink: #0F1619; --ink-2: #4C5A61;
    --ink-3: #77868D; --rule: #D5DDE0; --rule-hard: #B4C0C5;
    --s1: #0E8A75; --s2: #A94B26; --s3: #4C5A61;
    --mono: ui-monospace, "SF Mono", SFMono-Regular, Menlo, monospace;
    --sans: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  html, body {{ width: 1400px; height: 1000px; overflow: hidden; }}
  body {{ background: var(--paper); color: var(--ink); font-family: var(--sans);
    -webkit-font-smoothing: antialiased; padding: 54px 64px 42px;
    display: flex; flex-direction: column; }}
  .eyebrow {{ font-family: var(--mono); font-size: 15px; letter-spacing: 0.15em;
    text-transform: uppercase; color: var(--ink-3); }}
  h1 {{ font-family: var(--mono); font-weight: 600; font-size: 38px;
    letter-spacing: -0.02em; line-height: 1.18; margin: 13px 0 8px; }}
  h1 em {{ font-style: normal; color: var(--s1); }}
  .sub {{ font-size: 18.5px; color: var(--ink-2); max-width: 94ch; line-height: 1.5; }}
  .legend {{ display: flex; gap: 30px; margin: 20px 0 2px; font-family: var(--mono); font-size: 15.5px; }}
  .lg {{ display: flex; align-items: center; gap: 9px; white-space: nowrap; }}
  .sw {{ width: 26px; height: 4px; border-radius: 2px; }}
  .lg .fl {{ font-size: 13.5px; color: var(--ink-3); }}
  .plot {{ flex: 1; position: relative; margin-top: 4px; }}
  svg {{ width: 100%; height: 100%; display: block; overflow: visible; }}
  .grid {{ stroke: var(--rule); stroke-width: 1; }}
  .axis {{ stroke: var(--rule-hard); stroke-width: 2; }}
  .floorline {{ stroke-width: 2; stroke-dasharray: 7 6; opacity: 0.75; }}
  .ln {{ fill: none; stroke-width: 3; stroke-linejoin: round; stroke-linecap: round; }}
  .dot {{ stroke: var(--paper); stroke-width: 2.5; }}
  .tick {{ font-family: var(--mono); font-size: 14.5px; fill: var(--ink-3); }}
  .vlab {{ font-family: var(--mono); font-size: 14.5px; font-weight: 700; }}
  .axname {{ font-family: var(--mono); font-size: 14.5px; fill: var(--ink-3);
    letter-spacing: 0.09em; text-transform: uppercase; }}
  .note {{ font-family: var(--sans); font-size: 15px; fill: var(--ink-2); font-style: italic; }}
  .foot {{ font-family: var(--mono); font-size: 14.5px; color: var(--ink-3);
    border-top: 1px solid var(--rule-hard); padding-top: 14px; margin-top: 12px;
    display: flex; justify-content: space-between; gap: 30px; line-height: 1.5; }}
</style>
<body>
  <div class="eyebrow">Measured · 3 calls per point · each size on its own cold prefix · 31 Jul 2026</div>
  <h1>The bigger the context, the <em>less</em> cost scoring adds</h1>
  <div class="sub">Cost of 6 samples ÷ cost of 1 call, swept across context size. Every curve falls
    and flattens onto its floor — the point where the whole prompt rides the cache and only the
    answers regenerate.</div>

  <div class="legend">
    {legend}
    <div class="lg" style="color:var(--ink-3);font-size:13.5px">(dashed line = that model's floor)</div>
  </div>

  <div class="plot">
    <svg viewBox="0 0 {W} {H}" preserveAspectRatio="none">
      <g class="grid">
        {"".join(ygrid)}
      </g>
      {"".join(ytick)}
      <line class="axis" x1="{L}" y1="{TOP}" x2="{L}" y2="{BOT}"/>
      <line class="axis" x1="{L}" y1="{BOT}" x2="{R}" y2="{BOT}"/>
      {"".join(xtick)}
      <text class="axname" x="{(L + R) / 2}" y="{BOT + 57}" text-anchor="middle">context size (measured prompt tokens, log scale)</text>
      <text class="axname" x="30" y="{(TOP + BOT) / 2}" text-anchor="middle" transform="rotate(-90 30 {(TOP + BOT) / 2})">cost multiplier</text>

      {"".join(floorsvg)}

      {"".join(lines)}
      {ann}
    </svg>
  </div>

  <div class="foot">
    <span style="flex:1.2">Series don't align vertically because the x-axis is <b>measured tokens</b> —
      the same source text is<br>{tok_note} tokens at the smallest size. Anthropic's tokenizer runs ~54% longer.</span>
    <span style="flex:1;text-align:right">Gemini reaches the same 1.50× floor, just noisily — it caches in<br>
      8,192-token blocks, so the multiplier oscillates as the prefix crosses each boundary</span>
  </div>
</body>
"""
    with open(args.out, "w") as fh:
        fh.write(html)
    print(f"wrote {args.out}  ({sum(len(rows(m)) for m, _, _ in SERIES)} points plotted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
