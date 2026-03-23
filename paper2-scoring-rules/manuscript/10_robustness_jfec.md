# 8. Robustness

This section presents robustness checks for the main findings. We examine sensitivity to model specification, subsample stability, and alternative benchmarks.

## 8.1 Latent Dimension Sensitivity

Table 7 reports performance across different latent dimensions.

**Table 7: QLIKE Loss by Latent Dimension**

{{TABLE:latent_dim_sensitivity}}

*Notes: Panel-average QLIKE at H=1. Baseline specification uses d=4.*

The results show [sensitivity patterns to be inserted]. The choice of $d=4$ represents [justification to be inserted].

## 8.2 Subsample Stability

Table 8 compares performance across subperiods.

**Table 8: Performance by Subperiod**

{{TABLE:subperiod_performance}}

*Notes: 2020-2021 (crisis and recovery) vs. 2022-2024 (normalization).*

The shared regime advantage is [comparison to be inserted].

## 8.3 Asset Class Subsamples

Table 9 reports performance when the model is estimated separately for each asset class.

**Table 9: Within-Class vs. Cross-Class Pooling**

{{TABLE:pooling_scope}}

*Notes: "Within" pools only assets in the same class. "Cross" pools all 21 assets.*

This analysis tests whether regime structure is class-specific or truly cross-asset. The results suggest [findings to be inserted].

## 8.4 Alternative Observable Benchmarks

Table 10 compares the shared latent regime against additional observable transition variables.

**Table 10: Observable Benchmark Comparison**

{{TABLE:observable_benchmarks}}

*Notes: TED = TED spread. CDX = CDX North America IG. HY = High yield spread.*

The latent state outperforms all observable benchmarks by [magnitude to be inserted], confirming that [interpretation to be inserted].

## 8.5 Rolling vs. Expanding Window

Table 11 compares expanding window (baseline) with rolling window estimation.

**Table 11: Estimation Window Comparison**

{{TABLE:window_comparison}}

*Notes: Expanding uses all available history. Rolling uses 5-year window.*

The difference is [magnitude to be inserted], suggesting [interpretation to be inserted].

## 8.6 Forecast Combination

Table 12 examines whether combining shared and asset-specific forecasts improves performance.

**Table 12: Forecast Combination Results**

{{TABLE:forecast_combination}}

*Notes: Equal weighting and optimized weighting based on training period performance.*

Combination [does/does not] improve upon the best individual model, indicating [interpretation to be inserted].

## 8.7 Placebo Test: Shuffled Asset Panel

As a falsification check, we estimate the shared regime model on a shuffled panel where the time series order is randomized within each asset. If the model is capturing genuine cross-asset regime structure, shuffling should destroy this signal.

**Table 13: Placebo Test Results**

{{TABLE:placebo_shuffle}}

*Notes: Shuffled panel randomizes temporal ordering within each asset. 100 replications.*

The shuffled model performs [comparison to be inserted], confirming that [interpretation to be inserted].

## 8.8 Computational Considerations

Table 14 reports training and inference times.

**Table 14: Computational Cost**

{{TABLE:computation_times}}

*Notes: Training on NVIDIA A100 GPU. Inference per forecast iteration.*

The shared model is [comparison to be inserted] more expensive than asset-specific models, which may be a consideration for [applications to be inserted].

## 8.9 Summary

The robustness analysis supports the following conclusions:

1. Results are [stable/sensitive] to latent dimension choice within the range $d \in \{2, 4, 8\}$.

2. The shared regime advantage is [consistent/concentrated] across subperiods.

3. Cross-class pooling [does/does not] outperform within-class pooling, suggesting regime structure is [truly cross-asset / class-specific].

4. The latent state provides information beyond all tested observable benchmarks.

5. [Additional conclusions from robustness checks].
