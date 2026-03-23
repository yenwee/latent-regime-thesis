# Paper 2: When Does Regime Complexity Pay? Conditional Scoring Rules for Volatility Forecasts

**Author:** Yen Wee Lim, University of Malaya, Kuala Lumpur, Malaysia

## Abstract

Regime-switching volatility models are routinely evaluated by averaging loss functions unconditionally over the full out-of-sample period. We show that this practice can produce misleading model rankings when regime frequencies are imbalanced. We develop a regime-conditional evaluation framework that preserves the strict consistency of proper scoring rules while enabling assessment of model performance within each regime.

## Key Results

- **58.7% ranking reversal rate** across 63 asset-horizon pairs
- Regime-conditional MCS reveals "stress specialists" invisible to unconditional evaluation
- Framework extends to Fissler-Ziegel loss for joint VaR/ES evaluation
- Monte Carlo study: 33-48% misranking under calibrated DGPs

## Structure

```
paper2-shared-regimes/
├── scripts/
│   ├── simulation_study.py      # Monte Carlo (200 reps, 3 DGPs)
│   └── empirical_analysis.py    # 21-asset panel analysis
├── manuscript/
│   ├── sections/                # 6 sections (intro through conclusion)
│   └── build/                   # Makefile + pandoc build system
└── configs/
    └── default.yaml
```

## Building

```bash
cd manuscript/build && make pdf
```

## Related Papers

- **Paper 1** (Lim 2026a): Latent regime inference method (VRNN + STR-HAR)
- **Paper 3** (Lim 2026c): Economic validation of latent regimes
