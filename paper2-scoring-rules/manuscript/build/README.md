# Paper 2 Build System

Build system for "Cross-Asset Shared Latent Volatility Regimes" manuscript.

## Quick Start

```bash
cd manuscript/build
./build.sh              # Build PDF
make                    # Alternative: using Makefile
```

## Build Options

### Shell Script (`build.sh`)

```bash
./build.sh                              # Standard build
./build.sh --exp-dir ../../outputs/exp_v1  # With result injection
./build.sh --review                     # With line numbers
```

### Makefile

```bash
make all        # Build PDF (default)
make review     # Line-numbered version
make docx       # Word document
make latex      # LaTeX for Overleaf
make html       # HTML preview
make wordcount  # Word count
make clean      # Remove artifacts
```

## Section Structure

Paper 2 uses a different section structure than Paper 1:

| File | Section |
|------|---------|
| `03_introduction_jfec.md` | 1. Introduction |
| `04_related_work_jfec.md` | 2. Related Work |
| `05_conceptual_framework_jfec.md` | 3. Conceptual Framework |
| `06_model_jfec.md` | 4. Model |
| `07_identification_jfec.md` | 5. Identification & Interpretation |
| `08_data_jfec.md` | 6. Data & Experimental Design |
| `09_empirical_results_jfec.md` | 7. Empirical Results |
| `10_robustness_jfec.md` | 8. Robustness |
| `11_conclusion_jfec.md` | 9. Conclusion |
| `12_appendix_jfec.md` | Appendix |

## Placeholders

Results sections use placeholders that are replaced during build:

- `{{metric_name}}` - Single values (e.g., `{{latent_vix_corr:.2f}}`)
- `{{TABLE:table_name}}` - Full markdown tables (e.g., `{{TABLE:panel_qlike_summary}}`)

Run with `--exp-dir` to inject results from experiment output.

## Requirements

- pandoc
- xelatex (mactex-no-gui)
- Chicago CSL (auto-downloaded)

## Output

- `output/jfec_paper.pdf` - Main PDF
- `output/jfec_paper_review.pdf` - Line-numbered review version
