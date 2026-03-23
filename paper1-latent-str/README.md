# Deep-LSTR: Deep Latent Smooth-Transition HAR for Volatility Forecasting

This repository implements the Deep-LSTR framework for volatility forecasting, combining deep state-space models with Smooth Transition HAR specifications.

## Overview

Deep-LSTR uses a Variational Recurrent Neural Network (VRNN) to infer latent regime states, which then serve as transition variables in a Smooth Transition Regression framework. This approach captures regime dynamics that are difficult to observe directly from price data.

## Repository Structure

```
latent-str/
├── code.py                     # Original monolithic implementation (preserved)
├── configs/
│   ├── default.yaml            # Main configuration parameters
│   └── robustness.yaml         # Robustness check configurations
├── src/
│   ├── __init__.py             # Module exports
│   ├── utils.py                # Utilities, checkpointing, experiment management
│   ├── data.py                 # Data loading, feature prep, volatility estimators
│   ├── vrnn.py                 # Deep State-Space Model (VRNN)
│   ├── str_har.py              # Smooth Transition HAR with multiple transition functions
│   ├── garch.py                # GARCH family baselines
│   ├── metrics.py              # QLIKE, MSE, FZ0 loss functions
│   ├── dm_test.py              # Diebold-Mariano test
│   ├── mcs.py                  # Model Confidence Set
│   └── risk.py                 # VaR/ES evaluation
├── scripts/
│   ├── run_single_asset.py     # Run analysis for one asset
│   ├── run_panel.py            # Run full panel analysis
│   ├── run_robustness.py       # Run robustness checks (Section 6)
│   ├── generate_paper_figures.py  # Generate publication figures
│   └── make_tables.py          # Generate summary tables
├── outputs/                    # Results directory
│   ├── tables/                 # Summary tables
│   ├── figures/                # Plots
│   └── robustness/             # Robustness check results
├── drafts/                     # Paper drafts
│   └── jfec/                   # JFEC paper sections
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Single Asset Analysis

```bash
# Run for S&P 500 with 5-day horizon
python scripts/run_single_asset.py --ticker ^GSPC --horizon 5 --verbose

# Run for WTI Crude Oil with 1-day horizon
python scripts/run_single_asset.py --ticker CL=F --horizon 1 --verbose
```

### Full Panel Analysis

```bash
# Run all 21 assets across horizons 1, 5, 22
python scripts/run_panel.py --horizons 1 5 22 --n-jobs 3

# Use a specific experiment ID (for resumability)
python scripts/run_panel.py --exp-id baseline_v1

# Resume an interrupted experiment
python scripts/run_panel.py --exp-dir outputs/baseline_v1

# Generate summary tables
python scripts/make_tables.py --exp-dir outputs/baseline_v1

# Generate publication-quality figures
python scripts/generate_paper_figures.py --exp-dir outputs/baseline_v1
```

### Robustness Checks

```bash
# Run all robustness checks (Section 6 of the paper)
python scripts/run_robustness.py --all --n-jobs 4

# Run specific checks
python scripts/run_robustness.py --check latent_dims      # Table 7: d=1,2,4,8
python scripts/run_robustness.py --check transition_fns   # Table 8: logistic, exponential, double-logistic
python scripts/run_robustness.py --check volatility_estimators  # Table 10: GK, Parkinson, Rogers-Satchell
python scripts/run_robustness.py --check subsample        # Table 9: 2-year rolling windows

# Disable resumability (fresh run)
python scripts/run_robustness.py --check latent_dims --no-resume

# Transition function check with SSM sharing (default, much faster)
# Requires baseline run to exist with segment checkpoints
python scripts/run_robustness.py --check transition_fns --baseline-dir outputs/baseline_v1

# Force full SSM retraining for transition functions (slower)
python scripts/run_robustness.py --check transition_fns --no-share-ssm
```

**SSM Model Sharing (Transition Function Check)**: By default, the transition function robustness check reuses the latent state `q_ssm` from a baseline run instead of retraining the VRNN. This is valid because changing the transition function (logistic → exponential → double-logistic) only affects the STR-HAR regression, not the latent state inference. This optimization can reduce runtime by 80-90%.

## Configuration

### Main Configuration (`configs/default.yaml`)

```yaml
# Experiment settings
experiment:
  id: "exp_v1"          # Experiment ID (null = auto-generate timestamp)
  resume: true          # Enable checkpoint resumability

# Data parameters
data:
  start: "2015-01-01"
  end: "2025-12-31"
  volatility_estimator: "garman_klass"  # or "parkinson", "rogers_satchell"

# Deep SSM parameters
ssm:
  latent_dim: 2         # Latent state dimension
  epochs: 600           # Training epochs
  gru_hidden: 16        # GRU encoder hidden size
  decoder_hidden: 32    # MLP decoder hidden size

# STR-HAR parameters
str_har:
  gamma_max: 12.0       # Max transition steepness
  transition_fn: "logistic"  # or "exponential", "double_logistic"

# Rolling window
rolling:
  window: 600           # ~2.4 years
  retrain_freq: "A"     # Annual SSM retraining
```

### Robustness Configuration (`configs/robustness.yaml`)

Defines settings for the four robustness checks in Section 6:
- **6.1 Latent Dimensions**: `d ∈ {1, 2, 4, 8}`
- **6.2 Transition Functions**: logistic, exponential, double-logistic
- **6.3 Subsample Stability**: Rolling 2-year windows (2017-2024)
- **6.4 Volatility Estimators**: Garman-Klass, Parkinson, Rogers-Satchell

## Checkpoint Resumability

All scripts support checkpoint-based resumability for long-running experiments:

```bash
# If interrupted, just re-run the same command
python scripts/run_panel.py --exp-id my_experiment

# The system will:
# 1. Skip completed assets entirely
# 2. Resume SSM training from last completed segment
# 3. Resume forecasting from last checkpoint
```

Checkpoint structure:
```
outputs/my_experiment/
├── H1/
│   ├── checkpoints/
│   │   ├── GSPC_segment_0.pkl    # SSM segment checkpoints
│   │   ├── GSPC_segment_1.pkl
│   │   └── GSPC_forecasts.pkl    # Forecast progress
│   ├── GSPC_H1_results.csv       # Final results
│   └── GSPC_H1_mcs.csv           # MCS analysis
├── H5/
└── H22/
```

## Models

### Main Models
1. **HAR (OLS)**: Heterogeneous Autoregressive model - baseline
2. **STR-OBS**: Smooth Transition HAR with observable transition (smoothed log variance)
3. **STR-SSM (Deep-LSTR)**: Smooth Transition HAR with latent transition from Deep SSM

### Baseline Comparisons
- **GARCH(1,1)-t**: Standard GARCH with Student-t innovations
- **EGARCH(1,1)-t**: Exponential GARCH (asymmetric leverage)
- **MS-GARCH(2)-t**: Markov-Switching GARCH with 2 regimes

### Volatility Estimators
- **Garman-Klass (1980)**: Default, uses OHLC prices
- **Parkinson (1980)**: High-low range estimator
- **Rogers-Satchell (1991)**: Drift-adjusted OHLC estimator

### Transition Functions
- **Logistic**: Standard smooth transition (default, 9 parameters)
- **Exponential**: Faster symmetric transition (9 parameters)
- **Double-Logistic**: Two-threshold asymmetric transition (10 parameters)

## Evaluation Metrics

- **Point Forecasts**: MSE, QLIKE (quasi-likelihood)
- **Risk Forecasts**: FZ0 loss (proper elicitable scoring for VaR+ES)
- **Statistical Tests**:
  - Diebold-Mariano test with Newey-West HAC standard errors
  - Model Confidence Set (Hansen et al., 2011)

## Asset Universe (21 Assets)

| Category | Assets |
|----------|--------|
| **Equity Indices** | ^GSPC (S&P 500), ^NDX (Nasdaq 100), ^RUT (Russell 2000), ^DJI (Dow Jones) |
| **Equity Sectors** | XLF (Financials), XLK (Tech), XLE (Energy), XLU (Utilities) |
| **Rates/Duration** | ^IRX (3M T-Bill), ^TNX (10Y), ^TYX (30Y), IEF (7-10Y ETF), TLT (20+Y ETF) |
| **FX Majors** | EURUSD=X, USDJPY=X, GBPUSD=X, AUDUSD=X |
| **Commodities** | CL=F (WTI Oil), GC=F (Gold), NG=F (Natural Gas), HG=F (Copper) |

## Output Tables

The robustness script generates tables matching the paper format:

```
Table 7: Sensitivity to Latent Dimension
============================================================
Latent Dim (d)     QLIKE (H=5)     MCS Inclusion
----------------------------------------------------
1                  2.71            0.88
2 (baseline)       **2.65**        **0.94**
4                  2.68            0.91
8                  2.73            0.85
```

## Citation

If you use this code, please cite:

```bibtex
@article{deeplstr2025,
  title={Deep Latent Smooth-Transition Regime Identification for Volatility Forecasting},
  author={[Authors]},
  journal={Journal of Financial Econometrics},
  year={2025}
}
```

## License

MIT License
