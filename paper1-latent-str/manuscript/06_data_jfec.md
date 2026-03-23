# 4. Data and Empirical Design

## 4.1. Sample and Assets

Our sample spans January 2015 through December 2025, with out-of-sample evaluation beginning June 2017 using an expanding window design. We analyze 21 liquid financial assets across five asset classes:

**Equity Indices (4 assets):** ^GSPC (S&P 500), ^NDX (Nasdaq-100), ^RUT (Russell 2000), ^DJI (Dow Jones Industrial Average).

**Equity Sectors (4 assets):** XLF (Financials), XLK (Technology), XLE (Energy), XLU (Utilities).

**Rates and Duration (5 assets):** ^IRX (13-Week T-Bill Yield), ^TNX (10-Year Treasury Yield), ^TYX (30-Year Treasury Yield), IEF (7-10 Year Treasury ETF), TLT (20+ Year Treasury ETF).

**Foreign Exchange (4 assets):** EURUSD=X, USDJPY=X, GBPUSD=X, AUDUSD=X.

**Commodities (4 assets):** CL=F (WTI Crude Oil), GC=F (Gold), NG=F (Natural Gas), HG=F (Copper).

This design provides heterogeneity across volatility characteristics, liquidity profiles, and regime dynamics while maintaining sufficient sample size for robust inference. The multi-asset approach addresses concerns about specification overfitting that arise when models are developed on single assets, ensuring results generalize across diverse market microstructures.

## 4.2. Data Source and Realized Volatility Construction

Daily OHLC (Open, High, Low, Close) prices are obtained via the Yahoo Finance API. We employ the @garman1980estimation estimator for realized variance:
$$
\hat{\sigma}^2_{GK,t} = 0.5 \cdot [\ln(H_t/L_t)]^2 - (2\ln 2 - 1) \cdot [\ln(C_t/O_t)]^2
$$
where $H_t$, $L_t$, $O_t$, and $C_t$ denote daily high, low, open, and close prices respectively. The Garman-Klass estimator is approximately 7.4 times more efficient than the close-to-close squared return estimator under geometric Brownian motion, providing improved variance estimates from daily data. To ensure numerical stability, we floor realized variance at $\epsilon = 10^{-12}$ before log transformation.

Following @corsi2009simple, HAR regressors are constructed as:
- **Daily:** $RV^{(d)}_t = \ln(\hat{\sigma}^2_{GK,t})$
- **Weekly:** $RV^{(w)}_t = \frac{1}{5}\sum_{j=0}^{4} \ln(\hat{\sigma}^2_{GK,t-j})$
- **Monthly:** $RV^{(m)}_t = \frac{1}{22}\sum_{j=0}^{21} \ln(\hat{\sigma}^2_{GK,t-j})$

## 4.3. Sample Period and Regime Events

The out-of-sample period (June 2017–December 2025) encompasses multiple distinct volatility regimes providing rigorous stress-testing: the February 2018 "Volmageddon" event, the December 2018 Fed tightening selloff, the March 2020 COVID-19 crash when VIX reached 82, the 2022 aggressive Fed hiking cycle that generated unprecedented fixed income volatility, and the March 2023 regional banking crisis. This diverse event set ensures model evaluation spans both tranquil and turbulent market conditions.

## 4.4. Expanding Window Protocol

The deep state-space model is retrained annually on expanding windows to mitigate representation drift while preventing look-ahead bias:

| Retraining Date | Training Period | Evaluation Period |
|-----------------|-----------------|-------------------|
| 2017-06 | 2015-01 to 2017-06 | 2017-06 to 2017-12 |
| 2018-01 | 2015-01 to 2017-12 | 2018-01 to 2018-12 |
| ... | ... | ... |
| 2025-01 | 2015-01 to 2024-12 | 2025-01 to 2025-12 |

Input features are z-score standardized using statistics computed strictly on the expanding training window. Missing prices are forward-filled with a limit of 5 days; observations with zero or negative prices are excluded. Assets are aligned on the intersection of valid trading dates to ensure consistency across the panel.

At each forecast origin $t$, the model uses only information available at time $t$: (i) VRNN parameters trained exclusively on observations from the training window $[1, t_{train}]$, (ii) latent states inferred by forward-filtering through $[1, t]$ without smoothing, and (iii) STR-HAR coefficients estimated on the rolling window $[t-600, t]$. No future returns, volatilities, or test-period statistics inform any model parameter. This strict temporal separation ensures that reported out-of-sample performance reflects genuine forecasting ability rather than information leakage.

## 4.5. Data Availability

Replication code is publicly available at https://github.com/yenwee/latent-regime-thesis. The repository includes deterministic scripts to download all public data used in the analysis, along with a locked software environment for reproducibility.
