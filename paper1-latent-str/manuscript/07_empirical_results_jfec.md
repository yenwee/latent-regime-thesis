# 5. Empirical Results

## 5.1. Point Forecast Accuracy

Before presenting detailed forecast comparisons, we first illustrate the key mechanism underlying Deep-LSTR's advantage. Figure 1 demonstrates the regime smoothing mechanism through four complementary perspectives. Panel (A) shows the time series evolution of both transition functions for the S&P 500, revealing how the latent signal (Deep-LSTR, green) maintains smoother dynamics than the observable proxy (STR-HAR, blue). Panel (B) presents the distributional comparison, with the latent signal concentrated near the 0.5 threshold while the observable proxy shows a flatter distribution extending to extremes. Panel (C) provides a boxplot comparison highlighting the reduced dispersion of the latent approach, and Panel (D) summarizes the key statistics.

![](figures/fig1_regime_smoothing.pdf)

*Figure 1: Regime smoothing mechanism comparison for S&P 500 at H=1. Panel (A): Time series of transition function values G(s_t). Panel (B): Kernel density estimates. Panel (C): Distributional comparison with extreme value annotation. Panel (D): Summary statistics showing substantially lower volatility and extreme-day frequency for Deep-LSTR.*

Table 1 reports out-of-sample forecast accuracy across the 21-asset panel. We present results for the daily ($H=1$), weekly ($H=5$), and monthly ($H=22$) horizons using QLIKE loss, which is robust to heteroskedasticity in volatility proxies [@patton2011volatility].

**Table 1: Panel Summary of QLIKE Loss by Horizon**

| Model | H=1 | H=5 | H=22 |
|:---|:---:|:---:|:---:|
| HAR | -8.309 (0.298) | -8.624 (0.251) | -8.579 (0.250) |
| STR-OBS | -8.319 (0.299) | -8.644 (0.249) | -8.587 (0.249) |
| Deep-LSTR | -8.234 (0.329) | -8.636 (0.249) | -8.586 (0.249) |
| GARCH-t | -8.361 (0.254) | -8.414 (0.242) | -8.385 (0.242) |
| EGARCH-t | -8.375 (0.254) | -8.425 (0.242) | -8.383 (0.243) |
| MS-GARCH-t | -6.407 (0.795) | -7.000 (0.550) | -6.790 (0.840) |

*Notes:* Mean QLIKE loss across 19--20 assets (excluding IRX; GCF excluded at $H=1$ due to data quality). Newey-West standard errors in parentheses. Bold indicates lowest loss. Sample: June 2017--December 2025. Individual asset QLIKE values are winsorized to $[-20, 10]$ before aggregation to limit the influence of degenerate model fits (e.g., MS-GARCH on TNX/TYX).

The mean QLIKE results reveal a nuanced picture. At $H=1$, Deep-LSTR has the highest (worst) mean QLIKE among the RV-based models ($-8.234$ versus $-8.309$ for HAR and $-8.319$ for STR-OBS), driven by a single outlier asset (TNX, where the VRNN produces a poorly calibrated latent state at the daily horizon). However, examining *median* QLIKE---which is robust to such outliers---Deep-LSTR achieves the lowest or near-lowest median QLIKE at all horizons, with median improvements of 0.19% versus HAR and 0.17% versus STR-OBS at $H=1$, 0.08% versus HAR at $H=5$, and 0.02% versus HAR at $H=22$ (at $H=22$, STR-OBS achieves a negligibly lower median by 0.002 percentage points). The dominance on medians combined with the highest win rates (Table 2) confirms that Deep-LSTR improves forecasts for the large majority of assets, while a small number of outlier assets---where the latent state inference is poorly suited---disproportionately affect the mean.

Figure 2 illustrates the cumulative evolution of forecast performance over the out-of-sample period. The consistently positive cumulative QLIKE difference indicates that Deep-LSTR maintains its advantage throughout diverse market conditions, rather than concentrating gains in specific episodes.

![](figures/fig2_cumulative_qlike.pdf)

*Figure 2: Cumulative QLIKE loss difference over the out-of-sample period for S&P 500 at H=1. Panel (A) shows Deep-LSTR versus HAR; Panel (B) shows Deep-LSTR versus STR-HAR. Green regions indicate periods where Deep-LSTR outperforms. The consistently positive trajectory demonstrates sustained forecasting advantage across different market regimes.*

## 5.2. Win Rates and Model Confidence Sets

Table 2 reports the fraction of assets for which each model achieves the lowest QLIKE loss (Win Rate) and the fraction included in the 90% Model Confidence Set (MCS Inclusion).

**Table 2: Win Rates and MCS Inclusion by Horizon**

| Model | Win H=1 | MCS H=1 | Win H=5 | MCS H=5 | Win H=22 | MCS H=22 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| HAR | 0.00 | 0.68 | 0.00 | 0.20 | 0.00 | 0.15 |
| STR-OBS | 0.21 | 0.63 | 0.20 | 0.50 | 0.40 | 0.95 |
| Deep-LSTR | 0.63 | 0.89 | 0.80 | 0.90 | 0.60 | 1.00 |
| GARCH-t | 0.05 | 0.42 | 0.00 | 0.05 | 0.00 | 0.00 |
| EGARCH-t | 0.11 | 0.42 | 0.00 | 0.05 | 0.00 | 0.00 |
| MS-GARCH-t | 0.00 | 0.00 | 0.00 | 0.05 | 0.00 | 0.00 |

*Notes:* Win Rate is fraction of assets with lowest QLIKE. MCS Inclusion is fraction in 90% Model Confidence Set [@hansen2011model].

Deep-LSTR wins on 63% of assets at the daily horizon, 80% at the weekly horizon, and 60% at the monthly horizon. It is included in the MCS for 89% of assets at the daily horizon, 90% at the weekly horizon, and 100% at the monthly horizon, indicating robust performance that is rarely statistically dominated. By contrast, no GARCH-family model is included in the MCS at $H=22$, and even HAR achieves only 15% inclusion at the monthly horizon.

Figures 3 and 4 visualize these results. Figure 3 presents a heatmap showing the best-performing model for each asset-horizon combination among the RV-based specifications (HAR, STR-HAR, Deep-LSTR). The dominance of Deep-LSTR (green) across the panel is evident, with particularly strong performance for equity indices and commodities.

![](figures/fig3_winner_heatmap.pdf)

*Figure 3: Best model by asset and horizon among RV-based specifications. Green indicates Deep-LSTR achieves lowest QLIKE; blue indicates STR-HAR; gray indicates HAR. Horizontal lines separate asset classes.*

Figure 4 presents MCS inclusion rates across models, with the highlighted band showing Deep-LSTR's consistently high inclusion (89--100% across horizons). The dumbbell plot reveals that while STR-HAR shows moderate inclusion at longer horizons, Deep-LSTR is the only model that maintains robust inclusion across all horizons and assets.

![](figures/fig4_mcs_dumbbell.pdf)

*Figure 4: Model Confidence Set inclusion rates by horizon. Circle = H=1 (daily), square = H=5 (weekly), triangle = H=22 (monthly). The green highlighted band shows Deep-LSTR's consistently high inclusion rates. Dashed line indicates 50% inclusion threshold.*

## 5.3. Statistical Significance

Table 3 reports Diebold-Mariano test results for pairwise comparisons, showing the fraction of assets with significant differences at the 5% level.

**Table 3: Diebold-Mariano Rejection Rates**

| Comparison | H=1 | H=5 | H=22 |
|:---|:---:|:---:|:---:|
| Deep-LSTR vs. HAR | 0.21 | 0.85 | 0.70 |
| Deep-LSTR vs. STR-OBS | 0.26 | 0.50 | 0.00 |
| STR-OBS vs. HAR | 0.00 | 0.30 | 0.30 |

*Notes:* Fraction of 21 assets where Deep-LSTR significantly outperforms comparison model (DM test, p<0.05, Newey-West HAC standard errors with Bartlett kernel and fixed bandwidth $L = \max(20, 2H)$ where $H$ is the forecast horizon).

The latent transition variable provides statistically significant improvement over STR-OBS for 26% of assets at $H=1$ and 50% at $H=5$, though none at $H=22$ where both STR specifications converge. Against the HAR benchmark, Deep-LSTR achieves significant improvement for 21--85% of assets depending on horizon, with the strongest rejection rates at the weekly horizon. These results directly support the hypothesis that latent regime identification adds value beyond observable volatility proxies, particularly at the weekly forecasting horizon where regime dynamics are most informative.

## 5.4. Tail Risk Evaluation

Table 4 reports Expected Shortfall evaluation using the @fissler2016higher FZ0 loss function—the only valid scoring rule for joint VaR-ES comparison.

**Table 4a: FZ0 Loss by Coverage Level**

| Model | 1% | 5% |
|:---|:---:|:---:|
| HAR | -3.26 (0.17) | -3.72 (0.15) |
| STR-OBS | -3.23 (0.20) | -3.69 (0.16) |
| Deep-LSTR | **-3.31** (0.19) | **-3.76** (0.16) |
| EGARCH-t | -3.21 (0.15) | -3.69 (0.14) |

*Notes:* Mean FZ0 loss across 18--19 assets with cross-sectional standard errors in parentheses. TNX excluded from Deep-LSTR due to poorly calibrated latent state producing degenerate ES forecasts. Bold indicates lowest (best) loss.

**Table 4b: VaR Calibration (Violation Rate and Kupiec Test, @kupiec1995techniques)**

| Model | Viol 1% | Kupiec p (1%) | Viol 5% | Kupiec p (5%) |
|:---|:---:|:---:|:---:|:---:|
| HAR | 1.22 | 0.34 | 5.07 | 0.34 |
| STR-OBS | 1.35 | 0.27 | 5.39 | 0.29 |
| Deep-LSTR | 1.39 | 0.20 | 5.56 | 0.17 |
| EGARCH-t | 1.47 | 0.27 | 5.62 | 0.26 |

*Notes:* Mean violation rate (percent) and mean Kupiec $p$-value across 19 assets (excluding IRX; GCF excluded at $H=1$). Nominal coverage: 1% and 5%. Kupiec $p > 0.05$ indicates adequate calibration.

Deep-LSTR achieves the lowest FZ0 loss at both coverage levels while maintaining adequate calibration (Kupiec $p > 0.05$ for all models). The FZ0 improvement over HAR is 1.5% at the 1% VaR level and 1.1% at the 5% level. While modest in unconditional magnitude, the improvement is consistent across the panel and indicates tighter ES estimates without sacrificing VaR calibration.

An important interpretive caveat applies to these unconditional risk metrics. While the unconditional VaR calibration (Table 4b) indicates that all models provide adequate average coverage across the full sample, unconditional risk evaluation averages over both calm and turbulent periods. Because calm periods constitute approximately 75% of the sample, the unconditional metrics inherently dilute the economic magnitude of the latent regime's advantage during stress episodes. A model that provides substantially better tail coverage during high-volatility regimes—precisely when risk management matters most—may show only marginal unconditional improvement because its advantage is averaged with hundreds of calm-period observations where all models perform comparably. This structural limitation of unconditional scoring rules, which applies equally to QLIKE, FZ0, and VaR calibration tests, warrants separate methodological investigation. The regime-conditional evaluation framework necessary to properly quantify these within-regime differences is beyond the scope of this paper but represents a natural extension of the present work.

Figure 5 visualizes the FZ0 loss comparison across models, with error bars indicating cross-asset variation. Deep-LSTR achieves the lowest mean loss at both the 1% and 5% VaR levels, demonstrating consistent tail risk forecasting improvement across the panel.

![](figures/fig5_fz0_loss.pdf)

*Figure 5: Fissler-Ziegel FZ0 loss for joint VaR-ES evaluation. Panel (A) shows 1% VaR level; Panel (B) shows 5% VaR level. More negative values indicate better tail risk forecasts. Dashed green line indicates Deep-LSTR's mean loss. Error bars show cross-asset standard deviation.*

## 5.5. Regime Smoothing Mechanism

To understand why the latent transition variable improves forecasting, we examine its statistical properties relative to the observable proxy. Table 5 reports the distributional characteristics of the transition functions.

**Table 5: Transition Variable Properties**

| Property | STR-HAR | Deep-LSTR |
|:---|:---:|:---:|
| Standard Deviation | 0.28 | 0.213 |
| IQR | [0.28, 0.63] | [0.42, 0.55] |
| % Days at Extremes | 34.0% | 19.0% |
| Autocorrelation (lag-1) | 0.95 | 0.53 |

*Notes:* Statistics computed across 21-asset panel at H=1. Extremes defined as G < 0.2 or G > 0.8.

The latent transition variable exhibits substantially lower volatility and remains closer to the regime threshold ($G = 0.5$). While the observable proxy reaches extreme values on 34.0% of days, the latent signal does so on only 19.0% of days (Table 5). This "regime smoothing" property is associated with fewer false positive regime switches---days when the observable signal incorrectly indicates a regime change that does not persist. As illustrated in Figure 1 at the beginning of this section, our evidence suggests that the forecasting improvement is consistent with more conservative regime identification that filters transient volatility spikes, rather than early detection of regime shifts. The latent state appears to act as a de-noised regime indicator, blending regime-specific coefficients more smoothly and avoiding overreaction to short-lived volatility movements. We do not claim causal identification; rather, the evidence is consistent with a mechanism in which smoother regime identification reduces false positive switches. These findings support the hypothesis that Deep-LSTR achieves consistent, statistically significant gains despite modest daily win rates (~51%) because small improvements compound when false positive regime switches are systematically reduced.

## 5.6. Cross-Asset Heterogeneity

Table 6 examines where the latent regime approach provides greatest advantage.

**Table 6: Deep-LSTR Performance by Asset Class**

| Asset Class | QLIKE Improvement | Win Rate | MCS Inclusion | N |
|:---|:---:|:---:|:---:|:---:|
| Equity Indices | 0.0% | 1.00 | 1.00 | 4 |
| Equity Sectors | 0.0% | 1.00 | 1.00 | 4 |
| Fixed Income | 0.4% | 0.50 | 0.50 | 4 |
| Currencies | 0.0% | 0.75 | 1.00 | 4 |
| Commodities | 0.1% | 0.75 | 1.00 | 4 |

*Notes:* QLIKE Improvement relative to HAR baseline at H=5. Win Rate is fraction with lowest QLIKE. MCS Inclusion at 90% level.

The QLIKE improvements over HAR are modest in magnitude across all asset classes: fixed income shows the largest gain at 0.4%, followed by commodities at 0.1%, while equity indices, equity sectors, and currencies show negligible mean improvement (0.0%). However, the win rates and MCS inclusion rates tell a more compelling story. Deep-LSTR achieves 100% win rates for equity indices and sectors, 75% for currencies and commodities, and MCS inclusion of 100% for equities, currencies, and commodities, though only 50% for fixed income---where TNX and TYX favor the observable-transition specification at the weekly horizon (Table A2). This pattern---near-universal individual asset improvement with small magnitudes that wash out in averages---is consistent with the regime smoothing mechanism: the latent approach provides small but consistent gains across assets rather than large gains concentrated in specific markets.
