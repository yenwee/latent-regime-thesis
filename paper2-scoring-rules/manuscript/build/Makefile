# Paper 2: Regime-Conditional Scoring Rules - Build System
.DEFAULT_GOAL := all

SECTIONS_DIR := ../sections
BUILD_DIR := .
OUTPUT_DIR := output

SECTIONS := 01_introduction.md 02_theory.md 03_simulation.md 04_empirical.md 05_discussion.md 06_conclusion.md
SECTION_PATHS := $(addprefix $(SECTIONS_DIR)/,$(SECTIONS))

COMBINED := combined_paper.md
BIB := references.bib
CSL := chicago-author-date.csl
TEMPLATE := jfec-template.tex
OUTPUT := $(OUTPUT_DIR)/paper2_scoring_rules.pdf

PANDOC_OPTS := --from=markdown+yaml_metadata_block+tex_math_dollars+raw_tex+smart \
               --to=pdf \
               --pdf-engine=xelatex \
               --citeproc \
               --bibliography=$(BIB) \
               --csl=$(CSL) \
               --variable=geometry:margin=1in \
               --variable=linestretch:2 \
               --variable=fontsize:12pt

.PHONY: all
all: $(OUTPUT)
	@echo "Build complete: $(OUTPUT)"

$(OUTPUT_DIR):
	@mkdir -p $(OUTPUT_DIR)

$(COMBINED): paper.md $(SECTION_PATHS)
	@echo "Combining markdown sections..."
	@cat paper.md > $@
	@for section in $(SECTIONS); do \
		echo "" >> $@; \
		cat $(SECTIONS_DIR)/$$section >> $@; \
	done
	@echo "Created: $@ ($$(wc -w < $@) words)"

$(OUTPUT): $(COMBINED) $(BIB) $(CSL) $(TEMPLATE) | $(OUTPUT_DIR)
	@echo "Building PDF..."
	@pandoc $(COMBINED) $(PANDOC_OPTS) --template=$(TEMPLATE) --output=$@ 2>&1 | grep -v "^$$" || true
	@echo "Output: $@"

.PHONY: combined_paper.md
combined: $(COMBINED)

.PHONY: pdf
pdf: $(OUTPUT)

.PHONY: open
open: $(OUTPUT)
	@open $(OUTPUT)

.PHONY: wordcount
wordcount: $(COMBINED)
	@pandoc $(COMBINED) -t plain 2>/dev/null | wc -w | xargs echo "Total words:"

.PHONY: clean
clean:
	@rm -f $(COMBINED)
	@rm -rf $(OUTPUT_DIR)
	@echo "Cleaned"

.PHONY: rebuild
rebuild: clean all
