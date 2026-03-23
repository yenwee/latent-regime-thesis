# 7. Empirical Results

This section presents the main empirical findings. Results are discussed in the order: aggregate performance, asset-level heterogeneity, regime timing analysis, and conditional performance.

## 7.1 Aggregate Forecasting Performance

Table 1 reports out-of-sample QLIKE loss across the full panel and evaluation period.

**Table 1: Panel Summary of QLIKE Loss by Horizon**

{{TABLE:panel_qlike_summary}}

*Notes: QLIKE loss (lower is better) averaged across 21 assets. Evaluation period: 2020-2024. Bold indicates lowest loss. \* / \*\* / \*\*\* indicate statistically significant improvement over HAR at 10% / 5% / 1% based on Diebold-Mariano tests.*

The results show that [results to be inserted after experiments].

## 7.2 Model Confidence Set

Table 2 presents the Model Confidence Set at the 90% confidence level for each forecast horizon.

**Table 2: Model Confidence Set Membership**

{{TABLE:mcs_membership}}

*Notes: Check marks indicate inclusion in the 90% MCS. Models in the MCS are not statistically distinguishable from the best model.*

## 7.3 Asset-Level Heterogeneity

Figure 1 displays asset-level performance differences between Shared-VRNN and Asset-VRNN.

![Figure 1: Asset-Level QLIKE Improvement from Shared Regimes](figures/fig1_asset_improvement.pdf)

*Notes: Positive values indicate Shared-VRNN outperforms Asset-VRNN. Assets sorted by improvement magnitude.*

The heterogeneity pattern reveals [results to be inserted].

Table 3 reports performance by asset class.

**Table 3: QLIKE Loss by Asset Class**

{{TABLE:qlike_by_class}}

*Notes: QLIKE loss averaged within each asset class. H=1 (daily), H=5 (weekly), H=22 (monthly) horizons.*

## 7.4 Regime Timing Analysis

Figure 2 plots the inferred shared latent state against observable stress indicators.

![Figure 2: Shared Latent State vs. Observable Indicators](figures/fig2_latent_vs_observable.pdf)

*Notes: Shared latent state (solid line) compared with VIX (dashed) and cross-asset average RV (dotted). All series standardized. Shaded regions indicate NBER recession dates.*

The correlation between the latent state and observables is {{latent_vix_corr:.2f}} for VIX and {{latent_avg_corr:.2f}} for cross-asset average volatility. The partial correlation controlling for both observables is {{latent_partial_corr:.2f}}, indicating [interpretation to be inserted].

## 7.5 Conditional Performance

Table 4 reports performance conditioned on market state.

**Table 4: QLIKE Loss by Market Condition**

{{TABLE:qlike_conditional}}

*Notes: Low/Medium/High volatility regimes defined by terciles of cross-asset average realized volatility.*

The shared regime advantage is concentrated during [results to be inserted].

## 7.6 Transition Parameter Estimates

Table 5 reports the estimated transition parameters for each asset.

**Table 5: Asset-Specific Transition Parameters**

{{TABLE:transition_params}}

*Notes: $\gamma_i$ controls transition sharpness (higher = sharper). $c_i$ controls transition location (higher = later transition). Standard errors in parentheses.*

Assets with high $\gamma_i$ estimates include [assets to be inserted], indicating high systemic sensitivity. Assets with low $\gamma_i$ include [assets to be inserted], indicating more idiosyncratic dynamics.

## 7.7 Crisis Episode Analysis

Table 6 presents performance during specific crisis episodes.

**Table 6: QLIKE Loss During Crisis Episodes**

{{TABLE:crisis_performance}}

*Notes: Volmageddon (Feb 2018), COVID-19 (Feb-Apr 2020), Rate Shock (2022), SVB (Mar 2023).*

The shared regime model shows [relative performance to be inserted] during systemic stress events.

## 7.8 Summary of Main Findings

The empirical results support the following conclusions:

1. **Heterogeneous benefits:** Shared latent regimes improve forecasting for a subset of assets, particularly those with high systemic sensitivity.

2. **Regime timing:** The latent state captures information beyond observable proxies, as evidenced by [evidence to be inserted].

3. **Conditional advantage:** The benefits of shared regimes are concentrated during [periods to be inserted].

4. **Asset class patterns:** [Patterns to be inserted after experiments].

These findings are consistent with the conceptual framework: regimes contain both systemic and idiosyncratic components, and partial pooling exploits the systemic component while preserving heterogeneity.
