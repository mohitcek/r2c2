# post/

The published writing, frozen at publish state.

- `medium_post.md` — the article as published on Medium (2026-08-14):
  [Prompt Caching Makes Self-Consistency Cheap for Long-Context LLMs](https://medium.com/@mohitsinghchauhan/prompt-caching-makes-self-consistency-cheap-for-long-context-llms-c611ba4f5237)
- `linkedin_post.txt` — the LinkedIn post that announced it
- `medium_paste.html` — the rich-text source the article was pasted from (Medium has no markdown or table support; this page renders the post so headings, bold, links and code survive a copy-paste)
- `*_v1.png` — the designer-styled figure set used in the published article (`fig1_hero_1200_v1_fixed.png` is the one with the apostrophe typo patched); `fig7_cost_table.png` and `eq_cost_model.png` are the receipt table and the LaTeX-typeset equation, both rendered as images because Medium can't do tables or math
- `surface.gif` — the rotating cost surface embedded in the article (16:9, 20 s per revolution)

The figures are generated from the receipts by the scripts in `../experiments/figures/`; these are the exact bytes that went into the article.
