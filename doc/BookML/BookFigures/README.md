# BookFigures

Every figure appearing in the chapters of `doc/BookML`, one directory per
chapter, mirroring the layout of `BookPrograms`.

| Directory | Chapter | Figures |
|---|---|---|
| `chapter01_linear_algebra` | 1 — Linear Algebra | 5 |
| `chapter02_statistics` | 2 — Probability and Statistics | 5 |
| `chapter03_linear_regression` | 3 — Linear Regression | 4 |
| `chapter04_optimization` | 4 — Optimization | 4 |
| `chapter05_logistic_regression` | 5 — Logistic Regression | 4 |
| `chapter06_support_vector_machines` | 6 — Support Vector Machines | 4 |
| `chapter07_trees_and_ensembles` | 7 — Ensemble methods | 4 |

Each figure exists twice under the same base name: a `.pdf` for the LaTeX book
and a `.png` for the notebooks and the web.

## Regenerating

```bash
cd doc/BookML/BookFigures
python3 ch01_figures.py     # ... through ch07_figures.py
```

`common.py` holds the shared matplotlib settings and the `save()` helper, which
writes both formats. Every script is seeded, so the figures are reproducible and
reproduce the numbers quoted in the chapter text.

Figures are included from the chapters as

```latex
\includegraphics[width=0.66\textwidth]{BookFigures/chapter01_linear_algebra/name}
```

so `pdflatex` must be run from `doc/BookML`.
