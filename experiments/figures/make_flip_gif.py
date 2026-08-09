"""Build the coin-flip GIF: one question, six samples, six alternating verdicts.

Renders each reveal step as a frame with headless Chrome, then assembles them with
PIL. Flat colours compress well, so this lands well under the 5 MB both Medium and
LinkedIn allow.

    python experiments/figures/make_flip_gif.py --model Qwen/Qwen3.6-Plus
"""

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
SIZE = 1200


def verdict(answer):
    """Yes/No from the answer text. These all open with their verdict."""
    head = answer[:90].lower()
    if re.search(r"\b(not eligible|are not|no,|no —|ineligible)", head):
        return "No"
    if re.search(r"\b(yes|are eligible|is eligible)", head):
        return "Yes"
    return "?"


def snippet(answer, limit=135):
    text = re.sub(r"\s+", " ", answer).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut + "…"


def frame_html(question, verdicts, answers, revealed, score, show_score):
    chips = ""
    for i in range(len(verdicts)):
        if i < revealed:
            cls = "yes" if verdicts[i] == "Yes" else "no"
            label = verdicts[i]
            live = " live" if i == revealed - 1 else ""
        else:
            cls, label, live = "pending", "", ""
        chips += f'<span class="chip {cls}{live}">{label}</span>'

    quote = ""
    if 0 < revealed <= len(answers):
        quote = (
            f'<div class="quote"><span class="n">sample {revealed}</span>'
            f'{snippet(answers[revealed - 1])}</div>'
        )

    score_block = (
        f'<div class="score-wrap show"><div class="score">{score:.2f}</div>'
        f'<div class="score-k">noncontradiction<br>the only warning you get</div></div>'
        if show_score
        else '<div class="score-wrap"></div>'
    )

    return f"""<!doctype html>
<meta charset="utf-8">
<style>
  :root {{
    --paper:#F1F4F5; --ink:#0F1619; --ink-2:#4C5A61; --ink-3:#77868D;
    --rule:#D5DDE0; --rule-hard:#B4C0C5; --yes:#0E8A75; --no:#A94B26;
    --mono: ui-monospace,"SF Mono",SFMono-Regular,Menlo,monospace;
    --sans: system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  }}
  *{{margin:0;padding:0;box-sizing:border-box}}
  html,body{{width:{SIZE}px;height:{SIZE}px;overflow:hidden}}
  body{{background:var(--paper);color:var(--ink);font-family:var(--sans);
    -webkit-font-smoothing:antialiased;padding:66px 72px 56px;
    display:flex;flex-direction:column}}
  .eyebrow{{font-family:var(--mono);font-size:17px;letter-spacing:.15em;
    text-transform:uppercase;color:var(--ink-3)}}
  h1{{font-family:var(--mono);font-weight:600;font-size:52px;line-height:1.18;
    letter-spacing:-.02em;margin:18px 0 0}}
  .q{{font-size:29px;line-height:1.5;color:var(--ink-2);margin-top:22px;font-style:italic}}
  .q b{{color:var(--ink);font-style:normal}}
  .gap{{font-family:var(--mono);font-size:20px;color:var(--ink-3);margin-top:14px}}
  .chips{{display:flex;gap:16px;margin:52px 0 0}}
  .chip{{flex:1;height:132px;border-radius:6px;display:flex;align-items:center;
    justify-content:center;font-family:var(--mono);font-size:40px;font-weight:700;
    color:var(--paper)}}
  .chip.pending{{background:transparent;border:3px dashed var(--rule-hard)}}
  .chip.yes{{background:var(--yes)}}
  .chip.no{{background:var(--no)}}
  .chip.live{{outline:5px solid var(--ink);outline-offset:5px}}
  .quote{{margin-top:38px;font-size:25px;line-height:1.55;color:var(--ink-2);
    border-left:5px solid var(--rule-hard);padding-left:24px;min-height:150px}}
  .quote .n{{display:block;font-family:var(--mono);font-size:17px;
    letter-spacing:.12em;text-transform:uppercase;color:var(--ink-3);margin-bottom:7px}}
  .score-wrap{{flex:1;display:flex;align-items:center;gap:32px;min-height:170px}}
  .score-wrap.show .score{{font-family:var(--mono);font-weight:700;font-size:150px;
    letter-spacing:-.03em;color:var(--no);line-height:1}}
  .score-k{{font-family:var(--mono);font-size:22px;line-height:1.5;color:var(--ink-2)}}
  .foot{{border-top:2px solid var(--rule-hard);padding-top:18px;
    font-family:var(--mono);font-size:17px;color:var(--ink-3);line-height:1.5}}
</style>
<body>
  <div class="eyebrow">Same prompt · six samples · UQLM noncontradiction</div>
  <h1>One question. Six answers.</h1>
  <div class="q">{question}</div>
  <div class="gap">…the policy has a 24-hour rule and never says <b>who</b> cancelled.</div>
  <div class="chips">{chips}</div>
  {quote}
  {score_block}
  <div class="foot">Every answer fluent, confident, and citing the policy ·
    nothing in any single response reveals the disagreement</div>
</body>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--receipt", default="receipts/receipt_uq.json")
    ap.add_argument("--confidence", default="receipts/confidence_uq.json")
    ap.add_argument("--model", default="Qwen/Qwen3.6-Plus")
    ap.add_argument("--out", default="figures/flip.gif")
    args = ap.parse_args()

    receipt = json.load(open(args.receipt))
    run = next(r for r in receipt["runs"] if r["model"] == args.model)
    answers = [c["answer"] for c in run["calls"]]
    verdicts = [verdict(a) for a in answers]

    conf = json.load(open(args.confidence))
    score = next(m["noncontradiction_mean"] for m in conf["models"] if m["model"] == args.model)

    question = (
        "A non-refundable fare, booked <b>30 hours ago</b>. "
        "Then the <b>airline cancelled</b> the flight. Refund?"
    )
    print(f"{args.model}: {' '.join(verdicts)}  score {score:.2f}")

    # (revealed, show_score, duration_ms)
    steps = [(0, False, 900)]
    steps += [(i, False, 850) for i in range(1, len(answers) + 1)]
    steps += [(len(answers), True, 4200)]

    frames, durations = [], []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for idx, (revealed, show_score, ms) in enumerate(steps):
            html = tmp / f"f{idx}.html"
            png = tmp / f"f{idx}.png"
            html.write_text(frame_html(question, verdicts, answers, revealed, score, show_score))
            subprocess.run(
                [CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                 f"--window-size={SIZE},{SIZE}", f"--screenshot={png}", f"file://{html}"],
                check=True, capture_output=True,
            )
            frames.append(Image.open(png).convert("RGB"))
            durations.append(ms)
            print(f"  frame {idx + 1}/{len(steps)}")

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Flat colours, so a small palette costs nothing visually and keeps it light.
        palette = [f.quantize(colors=64, method=Image.MEDIANCUT) for f in frames]
        palette[0].save(
            out, save_all=True, append_images=palette[1:],
            duration=durations, loop=0, optimize=True, disposal=2,
        )

    kb = out.stat().st_size / 1024
    print(f"\nwrote {out}  {len(frames)} frames  {kb:.0f} KB")
    if kb > 5000:
        print("over the 5 MB platform limit — drop SIZE or colours")


if __name__ == "__main__":
    main()
