# JFEC Paper Build System

Converts markdown sections to PDF formatted for Journal of Financial Econometrics (Oxford University Press).

## Quick Start

```bash
# Install dependencies (macOS)
brew install pandoc
brew install --cask mactex-no-gui  # or mactex for full installation

# Build the PDF
cd drafts/jfec/build
./build.sh

# Or use make
make
```

## Directory Structure

```
drafts/jfec/
├── 01_title_page_jfec.md      # Title, authors, affiliations
├── 02_abstract_jfec.md        # Abstract (100 words max)
├── 03_introduction_jfec.md    # Introduction
├── 04_literature_review_jfec.md
├── 05_methodology_jfec.md
├── 06_data_jfec.md
├── 07_empirical_results_jfec.md
├── 08_robustness_jfec.md
├── 09_conclusion_jfec.md
├── 10_references_jfec.md      # Reference list (source for .bib)
├── 11_appendix_jfec.md
└── build/
    ├── build.sh               # Main build script
    ├── Makefile               # Alternative make-based build
    ├── defaults.yaml          # Pandoc configuration
    ├── jfec-template.tex      # LaTeX template (OUP style)
    ├── references.bib         # BibTeX bibliography
    ├── chicago-author-date.csl # Citation style
    ├── paper.md               # Master document with YAML metadata
    └── output/
        └── jfec_paper.pdf     # Generated PDF
```

## Build Options

### Standard Build
```bash
./build.sh
# or
make
```

### Review Version (with line numbers)
```bash
./build.sh --review
# or
make review
```

### Word Count
```bash
make wordcount
```

## JFEC Requirements Checklist

- [x] Maximum 40 pages (including all content)
- [x] Abstract under 100 words
- [x] Double-spaced for review
- [x] Chicago 15th edition citations
- [x] JEL classification codes
- [x] Significance stars: *, **, *** for 10%, 5%, 1%
- [x] Standard errors in parentheses

## Citation Usage

In markdown files, use citation keys from `references.bib`:

```markdown
As shown by @corsi2009simple, the HAR model...
The results align with previous findings [@hansen2011model; @patton2011volatility].
```

## Custom LaTeX Commands

The template provides these convenience commands:

```latex
% Significance stars
\onestar, \twostar, \threestar

% Math operators
\E, \Var, \Cov, \VaR, \ES, \QLIKE, \MSE, \MAE

% Realized volatility
\RV, \RVd, \RVw, \RVm

% Model names
\HAR, \STRHAR, \DeepLSTR, \VRNN, \GARCH
```

## Troubleshooting

### "pandoc not found"
```bash
brew install pandoc
```

### "xelatex not found"
```bash
brew install --cask mactex-no-gui
# Then restart terminal or:
eval "$(/usr/libexec/path_helper)"
```

### Font issues
The template uses Times New Roman. If not available, edit `jfec-template.tex`:
```latex
\setmainfont{TeX Gyre Termes}  % Free alternative
```

### Citation warnings
Ensure all `@citekey` references in markdown match entries in `references.bib`.
