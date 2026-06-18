# Defense Documentation

Four standalone LaTeX documents prepared as a reference/study set for the graduation defense.
Each compiles independently to its own PDF.

| File | Topic |
|------|-------|
| `01_dloc_architecture.tex` | The original DLoc network: CSI→heatmap input, ResNet encoder, location decoder, consistency/offset decoder, losses, training. |
| `02_dataset.tex` | The WILD dataset: MapFind collection, the two environments, the five Jacobs sessions, feature representation, and the train/test splits. |
| `03_model_architectures.tex` | Every model layer-by-layer (DLoc baseline, MobileNetV2, TinyCNN, MobileNetV2-UNet, Mamba) + knowledge-distillation setup. |
| `04_experiments.tex` | All six experiments (methodology, results, analysis) + a defense Q&A section. |

## Compile

Each document needs **two `pdflatex` passes** (for the table of contents and cross-references):

```bash
pdflatex 01_dloc_architecture.tex
pdflatex 01_dloc_architecture.tex
```

…and likewise for the other three. Or use `latexmk -pdf 04_experiments.tex`.

**Requirements:** a standard TeX distribution (TeX Live, MiKTeX) or **Overleaf**. Documents 1 and
4 use `tikz` / `pgfplots` (included in all standard distributions). No external image files are
needed — all figures are drawn in LaTeX.

## Suggested reading order
1 → 2 → 3 → 4 (architecture → data → models → experiments). Document 4 ends with a
"Defense preparation: anticipated questions" section worth rehearsing.
