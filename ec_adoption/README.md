# Paper draft: ACM EC 2026 (acmart acmsmall + acm-ec-26.sty)

Main file: `main.tex`. Compile with `latexmk -pdf main.tex`, or upload the zip to Overleaf (pdfLaTeX).
The cluster login node has no LaTeX.

The numbers and tables in `tables/` are generated from `results/tables/v2/` by `python scripts/paper_tables.py`,
which writes both this draft and the ICML draft in `../paper/`. Do not edit them by hand.

`template_sample/` holds the original EC sample files for reference. They are not part of the paper.

To-do notes (`\todo`, shown inline) mark pending results. References marked VERIFY in `refs.bib` still need checking.
