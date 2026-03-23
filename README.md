# Latent Regime Identification in Financial Volatility

A three-paper research program on latent regime-based volatility forecasting, evaluation, and economic validation.

**Author:** Yen Wee Lim, University of Malaya, Kuala Lumpur, Malaysia

## Papers

| Paper | Title | Pages | Status |
|-------|-------|-------|--------|
| 1 | [Latent State-Space Smooth Transition Models for Realized Volatility Forecasting](paper1-latent-str/manuscript/build/output/paper1_latent_str.pdf) | 55 | Complete |
| 2 | [When Does Regime Complexity Pay? Conditional Scoring Rules for Volatility Forecasts](paper2-scoring-rules/manuscript/build/output/paper2_scoring_rules.pdf) | 70 | Complete |
| 3 | [What Do Latent Volatility Regimes Represent? External Validation with Held-Out Stress Indicators](paper3-economic-validation/manuscript/build/output/paper3_economic_validation.pdf) | 68 | Complete |

## Research Arc

**Paper 1 (Applied):** Proposes Deep-LSTR, integrating Variational Recurrent Neural Networks with Smooth Transition HAR models. Evaluates 21 assets across equity, fixed income, currency, and commodity markets (2015-2025). Deep-LSTR wins on 60-80% of assets with MCS inclusion rates of 89-100%.

**Paper 2 (Methodological):** Develops regime-conditional proper scoring rules showing that standard unconditional QLIKE can misrank regime-switching models. Ranking reversals occur in 58.7% of asset-horizon pairs. Extends the framework to the Model Confidence Set and Fissler-Ziegel loss for joint VaR/ES evaluation.

**Paper 3 (Validation):** Tests whether latent regimes align with independent stress indicators not used in construction. Finds strong alignment (Cohen's d = 1.47 for VIX), 10-12 day credit spread lead, and 2-11 days faster crisis normalization than observable proxies.

## Repository Structure

```
latent-regime-thesis/
├── paper1-latent-str/           # Paper 1: Inference method
│   ├── src/                     # VRNN, STR-HAR, GARCH, metrics, evaluation
│   ├── scripts/                 # Panel run, robustness, figure generation
│   ├── configs/                 # Experiment configurations
│   └── manuscript/              # Markdown sections + build system
│
├── paper2-scoring-rules/       # Paper 2: Evaluation methodology
│   ├── scripts/                 # Simulation study, empirical analysis
│   └── manuscript/              # Theory sections + build system
│
├── paper3-economic-validation/  # Paper 3: Economic validation
│   ├── src/                     # Lead-lag, distributional, event study, regression
│   ├── scripts/                 # Validation runner, figure generation
│   └── manuscript/              # Results sections + build system
│
└── literature/                  # Reference papers
```

## Quick Start

```bash
# Paper 1: Full panel experiment (21 assets x 3 horizons)
cd paper1-latent-str
pip install -r requirements.txt
python scripts/run_panel.py --config configs/default.yaml

# Paper 2: Simulation + empirical analysis
cd paper2-scoring-rules
python scripts/simulation_study.py
python scripts/empirical_analysis.py

# Paper 3: External validation (requires Paper 1 outputs)
cd paper3-economic-validation
python scripts/run_validation.py
python scripts/generate_figures.py

# Build any paper's PDF
cd paper*/manuscript/build && make pdf
```

## Key Technical Features

- **Numba JIT acceleration:** GARCH/EGARCH/MS-GARCH inner loops (70-700x speedup)
- **Apple Silicon safe:** Lazy torch import + two-phase pipelined execution (avoids 400GB VSIZE)
- **Crash-safe:** Per-asset checkpointing with automatic resume
- **Data caching:** Parquet caching for yfinance/FRED downloads

## Requirements

- Python 3.10+, PyTorch, NumPy, Pandas, SciPy, statsmodels, yfinance, numba
- For PDFs: pandoc + xelatex (`brew install pandoc && brew install --cask mactex-no-gui`)

## License

This research is shared for academic purposes. Please cite appropriately if using any code or methodology.
