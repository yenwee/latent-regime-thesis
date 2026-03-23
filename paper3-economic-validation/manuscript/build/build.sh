#!/bin/bash
# =============================================================================
# JFEC Paper Build Script - Paper 3: Economic Validation of Latent Regimes
# Converts markdown sections to PDF for Journal of Financial Econometrics
# =============================================================================

set -e  # Exit on error

# Parse arguments
EXP_DIR=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --exp-dir)
            EXP_DIR="$2"
            shift 2
            ;;
        --review)
            REVIEW_MODE=true
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JFEC_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$SCRIPT_DIR"
OUTPUT_DIR="$BUILD_DIR/output"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Paper 3: Economic Validation Build${NC}"
echo -e "${GREEN}========================================${NC}"

# -----------------------------------------------------------------------------
# Step 0: Inject results (if results.yaml exists)
# -----------------------------------------------------------------------------
if [ -n "$EXP_DIR" ] && [ -f "$EXP_DIR/tables/results.yaml" ]; then
    echo -e "\n${YELLOW}Step 0: Injecting results from $EXP_DIR...${NC}"
    python "$PROJECT_ROOT/paper1-latent-str/scripts/inject_results.py" \
        --exp-dir "$EXP_DIR" \
        --source-dir "$JFEC_DIR" \
        --output "$BUILD_DIR/combined_paper.md" \
        --verbose
    INJECTION_DONE=true
    echo -e "${GREEN}Injection complete${NC}"
else
    echo -e "\n${YELLOW}Step 0: No results.yaml found, skipping injection${NC}"
    INJECTION_DONE=false
fi

# -----------------------------------------------------------------------------
# Step 1: Combine markdown files
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}Step 1: Combining markdown sections...${NC}"

COMBINED_MD="$BUILD_DIR/combined_paper.md"

# Start with YAML frontmatter from paper.md
head -n 20 "$BUILD_DIR/paper.md" > "$COMBINED_MD"

# Append each section (Paper 3 structure)
for section in \
    "03_introduction_jfec.md" \
    "04_conceptual_framing_jfec.md" \
    "05_economic_meaning_jfec.md" \
    "06_latent_measures_jfec.md" \
    "07_external_variables_jfec.md" \
    "08_empirical_strategy_jfec.md" \
    "09_case_studies_design_jfec.md" \
    "10_discussion_jfec.md" \
    "11_limitations_jfec.md" \
    "12_conclusion_jfec.md"
do
    if [ -f "$JFEC_DIR/$section" ]; then
        echo "" >> "$COMBINED_MD"
        tail -n +1 "$JFEC_DIR/$section" >> "$COMBINED_MD"
        echo -e "  ${GREEN}+${NC} $section"
    else
        echo -e "  ${RED}!${NC} $section (not found)"
    fi
done

echo -e "${GREEN}Combined file created: $COMBINED_MD${NC}"

# -----------------------------------------------------------------------------
# Step 2: Check dependencies
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}Step 2: Checking dependencies...${NC}"

check_command() {
    if command -v "$1" &> /dev/null; then
        echo -e "  ${GREEN}+${NC} $1 found"
        return 0
    else
        echo -e "  ${RED}!${NC} $1 not found"
        return 1
    fi
}

DEPS_OK=true
check_command pandoc || DEPS_OK=false
check_command xelatex || check_command pdflatex || DEPS_OK=false

if [ "$DEPS_OK" = false ]; then
    echo -e "\n${RED}Missing dependencies. Please install:${NC}"
    echo "  - pandoc: brew install pandoc"
    echo "  - LaTeX: brew install --cask mactex-no-gui"
    exit 1
fi

# -----------------------------------------------------------------------------
# Step 3: Download Chicago CSL if needed
# -----------------------------------------------------------------------------
CSL_FILE="$BUILD_DIR/chicago-author-date.csl"
if [ ! -f "$CSL_FILE" ]; then
    echo -e "\n${YELLOW}Step 3: Downloading Chicago CSL style...${NC}"
    curl -sL "https://raw.githubusercontent.com/citation-style-language/styles/master/chicago-author-date.csl" \
        -o "$CSL_FILE"
    echo -e "  ${GREEN}+${NC} Downloaded chicago-author-date.csl"
else
    echo -e "\n${YELLOW}Step 3: Chicago CSL already exists${NC}"
fi

# -----------------------------------------------------------------------------
# Step 4: Build PDF
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}Step 4: Building PDF...${NC}"

cd "$BUILD_DIR"

# Run pandoc with the template
pandoc "$COMBINED_MD" \
    --from=markdown+yaml_metadata_block+tex_math_dollars+raw_tex+smart \
    --to=pdf \
    --pdf-engine=xelatex \
    --template=jfec-template.tex \
    --citeproc \
    --bibliography=references.bib \
    --csl=chicago-author-date.csl \
    --variable=geometry:margin=1in \
    --variable=linestretch:2 \
    --variable=fontsize:12pt \
    --output="$OUTPUT_DIR/jfec_paper.pdf" \
    2>&1 | tee "$OUTPUT_DIR/build.log"

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}Build successful!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "Output: $OUTPUT_DIR/jfec_paper.pdf"

    # Show page count if possible
    if command -v pdfinfo &> /dev/null; then
        PAGES=$(pdfinfo "$OUTPUT_DIR/jfec_paper.pdf" | grep Pages | awk '{print $2}')
        echo -e "Pages: $PAGES (JFEC limit: 40)"
    fi
else
    echo -e "\n${RED}Build failed. Check $OUTPUT_DIR/build.log for details.${NC}"
    exit 1
fi

# -----------------------------------------------------------------------------
# Step 5: Optional - Build review version with line numbers
# -----------------------------------------------------------------------------
if [ "$REVIEW_MODE" = true ]; then
    echo -e "\n${YELLOW}Building review version with line numbers...${NC}"

    # Create modified template with line numbers enabled
    sed 's/% \\linenumbers/\\linenumbers/' jfec-template.tex > jfec-template-review.tex

    pandoc "$COMBINED_MD" \
        --from=markdown+yaml_metadata_block+tex_math_dollars+raw_tex+smart \
        --to=pdf \
        --pdf-engine=xelatex \
        --template=jfec-template-review.tex \
        --citeproc \
        --bibliography=references.bib \
        --csl=chicago-author-date.csl \
        --variable=geometry:margin=1in \
        --variable=linestretch:2 \
        --variable=fontsize:12pt \
        --output="$OUTPUT_DIR/jfec_paper_review.pdf"

    rm -f jfec-template-review.tex
    echo -e "${GREEN}Review version: $OUTPUT_DIR/jfec_paper_review.pdf${NC}"
fi

echo -e "\n${GREEN}Done!${NC}"
