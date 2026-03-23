# 6. Data and Experimental Design

This section describes the data, evaluation methodology, and experimental design. The asset panel and time period match our companion paper on single-asset regimes, enabling direct comparison of shared versus asset-specific approaches.

## 6.1 Asset Panel

We construct a panel of 21 assets spanning five asset classes:

**Equity indices (4):** S&P 500, NASDAQ-100, Russell 2000, Dow Jones Industrial Average. These represent large-cap, technology-weighted, small-cap, and blue-chip U.S. equity exposures.

**Fixed income (4):** 2-Year, 5-Year, 10-Year, and 30-Year U.S. Treasury futures. These span the yield curve and capture interest rate volatility at different maturities.

**Currencies (6):** EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CAD, USD/CHF. Major currency pairs against the U.S. dollar representing developed market exchange rate dynamics.

**Commodities (5):** Gold, Silver, Crude Oil (WTI), Natural Gas, Copper. Precious metals, energy, and industrial metals representing commodity market volatility.

**Alternatives (2):** Bitcoin and Ethereum. Cryptocurrency assets that have become increasingly integrated with traditional markets.

The asset selection balances coverage across asset classes with data availability and liquidity. All assets have active futures or spot markets with intraday data sufficient for realized volatility construction.

## 6.2 Realized Volatility Construction

Daily realized volatility is computed from 5-minute intraday returns:
$$
RV_t = \sum_{i=1}^{M} r_{t,i}^2
$$
where $r_{t,i}$ is the $i$-th intraday return and $M$ is the number of intraday observations (typically 78 for U.S. equity hours, varying by asset class).

For assets with different trading hours, we use the primary trading session to ensure comparability. Pre- and post-market returns are aggregated into the first and last intraday intervals, respectively.

We apply standard data cleaning: removal of overnight gaps, treatment of trading halts, and winsorization of extreme observations at the 0.1% and 99.9% percentiles. The log transformation $\log(RV_t)$ is used for modeling, with results reported in volatility levels for interpretability.

## 6.3 Sample Period

The sample spans January 2015 through December 2024, providing 10 years of daily observations. This period includes several distinct market regimes:

- **2015-2016:** Low volatility period with brief China-related turbulence
- **2017-2018:** Continued low volatility followed by the "Volmageddon" episode
- **2019:** Trade war uncertainty with elevated but not extreme volatility
- **2020:** COVID-19 market disruption with historically extreme volatility
- **2021-2022:** Recovery followed by rate shock volatility
- **2023-2024:** Normalization with episodic stress (banking crisis, geopolitical events)

The diversity of regimes provides a strong test of the model's ability to identify and exploit regime structure.

## 6.4 Training and Evaluation Split

We use an expanding window evaluation design. The initial training period covers 2015-2019 (5 years). We then produce out-of-sample forecasts for 2020-2024, retraining quarterly to incorporate new information.

This design tests the model's ability to generalize to regime conditions not seen in training. The 2020 COVID-19 episode is particularly important: models trained on 2015-2019 data must forecast through an unprecedented volatility event.

## 6.5 Forecast Horizons

We evaluate forecasts at three horizons:

- **Daily ($h=1$):** One-day-ahead volatility forecasts
- **Weekly ($h=5$):** Five-day-ahead forecasts
- **Monthly ($h=22$):** Twenty-two-day-ahead forecasts

These horizons correspond to common risk management and portfolio rebalancing frequencies.

## 6.6 Evaluation Metrics

Forecast performance is evaluated using the QLIKE loss function:
$$
\text{QLIKE}(\hat{\sigma}^2, \sigma^2) = \log(\hat{\sigma}^2) + \frac{\sigma^2}{\hat{\sigma}^2}
$$
where $\hat{\sigma}^2$ is the forecast and $\sigma^2$ is realized volatility. QLIKE is a proper scoring rule for volatility that penalizes both under- and over-prediction, with stronger penalties for under-prediction.

We also report MSE for comparability with prior literature, though QLIKE is our primary metric given its robustness to proxy noise [@patton2011volatility].

## 6.7 Statistical Testing

We use the Model Confidence Set (MCS) procedure [@hansen2011model] to identify the set of models that are not significantly outperformed by others. The MCS provides a multiple comparison framework that accounts for the joint distribution of forecast errors.

For pairwise comparisons, we use the Diebold-Mariano test [@diebold1995comparing] with heteroskedasticity and autocorrelation consistent standard errors.

## 6.8 Model Comparisons

The following models are compared:

1. **Shared-VRNN:** The proposed multivariate VRNN with shared latent regimes and asset-specific STR-HAR
2. **Asset-VRNN:** Single-asset VRNN with asset-specific latent regimes (companion paper model)
3. **STR-HAR (VIX):** STR-HAR with VIX as the transition variable
4. **STR-HAR (PCA):** STR-HAR with first PC of the volatility panel as transition variable
5. **STR-HAR (Avg):** STR-HAR with cross-asset average volatility as transition variable
6. **HAR:** Standard HAR model without regime structure

These comparisons isolate the contribution of:
- Cross-asset information (Shared-VRNN vs. Asset-VRNN)
- Latent vs. observable regimes (Shared-VRNN vs. STR-HAR with observables)
- Regime structure itself (all STR models vs. HAR)

## 6.9 Robustness Checks

We conduct several robustness analyses:

**Asset class subsamples:** Separate evaluation for equity, fixed income, currency, and commodity assets to assess heterogeneity in model performance.

**Subsample stability:** Evaluation over 2020-2021 (crisis and recovery) vs. 2022-2024 (normalization) to assess performance across different market conditions.

**Alternative latent dimensions:** Sensitivity to the choice of latent dimension $d \in \{2, 4, 8\}$.

**Alternative regime benchmarks:** Comparison with additional observable regime indicators including credit spreads and funding liquidity measures.
