# Paper 3: What Do Latent Volatility Regimes Represent? External Validation with Held-Out Stress Indicators

**Author:** Yen Wee Lim, University of Malaya, Kuala Lumpur, Malaysia

## Abstract

This paper examines whether latent volatility regimes exhibit external economic coherence --- whether regime states, constructed from realized volatility data alone, align systematically with independent measures of market stress not used in their construction.

## Key Results

- **Cohen's d = 1.47** for VIX distributional separation across regime states
- Latent regime **leads credit spreads by 10-12 trading days** (mediated through volatility channel)
- **15-day early activation** before COVID crash peak
- **2-11 days faster normalization** than observable proxy across all 4 crisis episodes
- Term spread prediction significant at all horizons after controlling for RV (p < 0.013)

## Structure

```
paper3-economic-validation/
├── src/
│   ├── external_data.py         # Yahoo/FRED stress proxy fetching
│   ├── regime_loader.py         # Load Paper 1 regime outputs
│   ├── lead_lag.py              # Cross-correlation + Granger causality
│   ├── distributional.py        # KS, Mann-Whitney, block bootstrap
│   ├── event_study.py           # Crisis episode analysis
│   └── predictive_regression.py # OLS with Newey-West HAC
├── scripts/
│   ├── run_validation.py        # Main analysis runner
│   └── generate_figures.py      # 8 publication figures
├── manuscript/
│   └── build/                   # Makefile + pandoc build
└── configs/
    └── default.yaml
```

## Building

```bash
python scripts/run_validation.py    # ~4 seconds
python scripts/generate_figures.py  # generates 8 PDFs
cd manuscript/build && make pdf
```

## Related Papers

- **Paper 1** (Lim 2026a): Latent regime inference method (VRNN + STR-HAR)
- **Paper 2** (Lim 2026b): Regime-conditional scoring rules for evaluation
