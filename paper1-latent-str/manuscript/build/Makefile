# JFEC Paper Makefile
# Converts markdown to PDF for Journal of Financial Econometrics
# Usage: make [target]

.DEFAULT_GOAL := all

# Directories
JFEC_DIR := ..
BUILD_DIR := .
OUTPUT_DIR := output
FIGURES_DIR := $(JFEC_DIR)/figures
EXP_DIR := ../../../data/exp_20260115_124219

# Files
SECTIONS := 03_introduction_jfec.md 04_literature_review_jfec.md 05_methodology_jfec.md \
            06_data_jfec.md 07_empirical_results_jfec.md 08_robustness_jfec.md \
            09_conclusion_jfec.md 10_references_jfec.md 11_appendix_jfec.md
SECTION_PATHS := $(addprefix $(JFEC_DIR)/,$(SECTIONS))

COMBINED := combined_paper.md
BIB := references.bib
CSL := chicago-author-date.csl
TEMPLATE := jfec-template.tex
OUTPUT := $(OUTPUT_DIR)/paper1_latent_str.pdf

# Pandoc common options (no --number-sections since markdown has manual numbering)
PANDOC_OPTS := --from=markdown+yaml_metadata_block+tex_math_dollars+raw_tex+smart \
               --to=pdf \
               --pdf-engine=xelatex \
               --citeproc \
               --bibliography=$(BIB) \
               --csl=$(CSL) \
               --variable=geometry:margin=1in \
               --variable=linestretch:2 \
               --variable=fontsize:12pt

# Default target
.PHONY: all
all: $(OUTPUT)
	@echo "Build complete: $(OUTPUT)"

# Create output directory
$(OUTPUT_DIR):
	@mkdir -p $(OUTPUT_DIR)

# Combine markdown files
$(COMBINED): paper.md $(SECTION_PATHS)
	@echo "Combining markdown sections..."
	@head -n 14 paper.md > $@
	@for section in $(SECTIONS); do \
		echo "" >> $@; \
		cat $(JFEC_DIR)/$$section >> $@; \
	done
	@echo "Created: $@"

# Build PDF
$(OUTPUT): $(COMBINED) $(BIB) $(CSL) $(TEMPLATE) | $(OUTPUT_DIR)
	@echo "Building PDF..."
	@pandoc $(COMBINED) $(PANDOC_OPTS) --template=$(TEMPLATE) --output=$@ 2>&1 | grep -v "^$$" || true
	@echo "Output: $@"

# Review version with line numbers
.PHONY: review
review: $(COMBINED) $(BIB) $(CSL) $(TEMPLATE) | $(OUTPUT_DIR)
	@echo "Building review version with line numbers..."
	@sed 's/% \\linenumbers/\\linenumbers/' $(TEMPLATE) > jfec-template-review.tex
	@pandoc $(COMBINED) $(PANDOC_OPTS) --template=jfec-template-review.tex --output=$(OUTPUT_DIR)/jfec_paper_review.pdf 2>&1 | grep -v "^$$" || true
	@rm -f jfec-template-review.tex
	@echo "Review output: $(OUTPUT_DIR)/jfec_paper_review.pdf"

# Word count (approximate, excluding references)
.PHONY: wordcount
wordcount: $(COMBINED)
	@echo "Word count (approximate):"
	@pandoc $(COMBINED) -t plain 2>/dev/null | wc -w | xargs echo "  Total words:"
	@echo "  JFEC limit: ~6500 words (excl. references/tables)"

# Page count
.PHONY: pagecount
pagecount: $(OUTPUT)
	@if command -v pdfinfo >/dev/null 2>&1; then \
		pages=$$(pdfinfo $(OUTPUT) 2>/dev/null | grep Pages | awk '{print $$2}'); \
		echo "Page count: $$pages (JFEC limit: 40)"; \
	else \
		echo "Install poppler for page count: brew install poppler"; \
	fi

# Watch for changes and rebuild (requires fswatch)
.PHONY: watch
watch:
	@echo "Watching for changes... (Ctrl+C to stop)"
	@fswatch -o $(JFEC_DIR)/*.md paper.md $(TEMPLATE) | xargs -n1 -I{} make all

# Open PDF
.PHONY: open
open: $(OUTPUT)
	@open $(OUTPUT)

# Build DOCX (Word document)
.PHONY: docx
docx: $(COMBINED) $(BIB) $(CSL) | $(OUTPUT_DIR)
	@echo "Building DOCX..."
	@pandoc $(COMBINED) \
		--from=markdown+yaml_metadata_block+tex_math_dollars+raw_tex+smart \
		--to=docx \
		--citeproc \
		--bibliography=$(BIB) \
		--csl=$(CSL) \
		--output=$(OUTPUT_DIR)/jfec_paper.docx 2>&1 | grep -v "^$$" || true
	@echo "DOCX output: $(OUTPUT_DIR)/jfec_paper.docx"
	@open $(OUTPUT_DIR)/jfec_paper.docx

# Build LaTeX source (for Overleaf or manual editing)
.PHONY: latex
latex: $(COMBINED) $(BIB) $(CSL) $(TEMPLATE) | $(OUTPUT_DIR)
	@echo "Building LaTeX..."
	@pandoc $(COMBINED) $(PANDOC_OPTS) \
		--to=latex \
		--template=$(TEMPLATE) \
		--standalone \
		--output=$(OUTPUT_DIR)/jfec_paper.tex
	@cp $(BIB) $(OUTPUT_DIR)/
	@echo "LaTeX output: $(OUTPUT_DIR)/jfec_paper.tex"
	@echo "Bibliography: $(OUTPUT_DIR)/references.bib"
	@echo "Upload both to Overleaf for editing"

# Build HTML preview
.PHONY: html
html: $(COMBINED) $(BIB) $(CSL) | $(OUTPUT_DIR)
	@echo "Building HTML..."
	@pandoc $(COMBINED) \
		--from=markdown+yaml_metadata_block+tex_math_dollars+raw_tex+smart \
		--to=html5 \
		--standalone \
		--mathjax \
		--citeproc \
		--bibliography=$(BIB) \
		--csl=$(CSL) \
		--toc \
		--css=https://cdn.jsdelivr.net/npm/water.css@2/out/water.css \
		--output=$(OUTPUT_DIR)/jfec_paper.html
	@echo "HTML output: $(OUTPUT_DIR)/jfec_paper.html"
	@open $(OUTPUT_DIR)/jfec_paper.html

# Generate figures from experiment data
.PHONY: figures
figures:
	@echo "Generating figures from experiment data..."
	@python $(FIGURES_DIR)/generate_figures.py --exp-dir $(EXP_DIR)
	@echo "Figures generated in: $(FIGURES_DIR)"

# Build Beamer presentation slides
.PHONY: slides
slides: $(COMBINED) $(BIB) $(CSL) | $(OUTPUT_DIR)
	@echo "Building Beamer slides..."
	@pandoc $(COMBINED) \
		--from=markdown+yaml_metadata_block+tex_math_dollars+raw_tex+smart \
		--to=beamer \
		--pdf-engine=xelatex \
		--citeproc \
		--bibliography=$(BIB) \
		--csl=$(CSL) \
		--slide-level=2 \
		--variable=theme:metropolis \
		--variable=aspectratio:169 \
		--output=$(OUTPUT_DIR)/jfec_slides.pdf 2>&1 | grep -v "^$$" || true
	@echo "Slides output: $(OUTPUT_DIR)/jfec_slides.pdf"
	@open $(OUTPUT_DIR)/jfec_slides.pdf

# Clean up
.PHONY: clean
clean:
	@rm -f $(COMBINED)
	@rm -rf $(OUTPUT_DIR)
	@rm -f *.aux *.log *.out *.toc *.bbl *.blg jfec-template-review.tex
	@echo "Cleaned build artifacts"

# Full rebuild
.PHONY: rebuild
rebuild: clean all

# Help
.PHONY: help
help:
	@echo "JFEC Paper Build System"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all       - Build the PDF (default)"
	@echo "  figures   - Generate figures from experiment data"
	@echo "  docx      - Build Word document (.docx)"
	@echo "  latex     - Build LaTeX source for Overleaf"
	@echo "  html      - Build HTML preview"
	@echo "  slides    - Build Beamer presentation slides"
	@echo "  review    - Build review version with line numbers"
	@echo "  open      - Build and open the PDF"
	@echo "  watch     - Watch for changes and auto-rebuild"
	@echo "  wordcount - Show approximate word count"
	@echo "  pagecount - Show page count (requires poppler)"
	@echo "  clean     - Remove generated files"
	@echo "  rebuild   - Clean and rebuild"
	@echo "  help      - Show this help"
	@echo ""
	@echo "Requirements:"
	@echo "  - pandoc (brew install pandoc)"
	@echo "  - xelatex (brew install --cask mactex-no-gui)"
	@echo ""
	@echo "Optional:"
	@echo "  - poppler (brew install poppler) for page count"
	@echo "  - fswatch (brew install fswatch) for watch mode"
