# 6. Empirical Strategy

This section describes the empirical approach for assessing alignment between latent regimes and external variables. The strategy emphasizes transparent, pre-specified methods that minimize researcher degrees of freedom. All analyses are designed before examining results; the methods are documented here as a commitment device.

## 6.1 Lead-Lag Analysis

The primary tool for examining temporal relationships is cross-correlation analysis between the regime indicator and external variables at various leads and lags.

### 6.1.1 Specification

For regime indicator $q_t$ and external variable $x_{j,t}$, we compute:

$$\rho_j(k) = \text{Corr}(q_t, x_{j,t+k})$$

for $k \in \{-K, \ldots, 0, \ldots, K\}$, where $K$ is the maximum lead/lag considered. Positive $k$ corresponds to the regime leading the external variable; negative $k$ corresponds to the regime lagging.

The baseline specification uses $K = 20$ trading days, corresponding to approximately four weeks. This window is sufficiently wide to capture meaningful lead-lag relationships while avoiding spurious correlations at extreme lags.

### 6.1.2 Inference

Standard errors for cross-correlations account for serial dependence in both series. We use @newey1987simple standard errors with automatic bandwidth selection to accommodate unknown persistence structures. The null hypothesis is zero correlation at each lag.

We do not apply multiple testing corrections across lags within a single variable. The cross-correlation function is inherently a joint object; correcting for multiple comparisons would be overly conservative and obscure economically meaningful patterns. However, we apply appropriate corrections when comparing across external variables.

### 6.1.3 Interpretation

Lead-lag patterns are informative about temporal precedence but do not establish causation. A regime that leads funding stress by several days could reflect:

- The regime anticipating funding deterioration
- Both responding to a common unobserved factor with different speeds
- Spurious correlation due to shared low-frequency trends

We interpret lead-lag evidence as characterizing alignment, not identifying mechanisms.

## 6.2 Distributional Comparison Across Regimes

The second component examines whether external variables exhibit different distributions in high-regime versus low-regime periods.

### 6.2.1 Regime Classification

For distributional analysis, we classify each observation into high or low regime based on the regime indicator:

$$\text{High regime: } q_t > 0.5 \qquad \text{Low regime: } q_t \leq 0.5$$

The threshold of 0.5 corresponds to equal posterior probability of the two states. Alternative thresholds (0.3, 0.7) are examined in robustness analysis.

### 6.2.2 Comparison Statistics

For each external variable $x_j$, we compute:

**Means.** The difference in conditional means:

$$\Delta_j = \mathbb{E}[x_{j,t} | q_t > 0.5] - \mathbb{E}[x_{j,t} | q_t \leq 0.5]$$

with standard errors computed via block bootstrap to account for temporal dependence.

**Distributions.** We compare full distributions using Kolmogorov-Smirnov tests, again with bootstrap inference to accommodate dependence [@politis1994stationary]. Quantile-quantile plots provide visual assessment of distributional differences.

**Variance ratios.** We examine whether external variables are more volatile in high-regime periods:

$$\text{VR}_j = \frac{\text{Var}(x_{j,t} | q_t > 0.5)}{\text{Var}(x_{j,t} | q_t \leq 0.5)}$$

### 6.2.3 Expected Patterns

If regimes capture stress-related variation, we would expect:

- Higher funding spreads in high-regime periods ($\Delta_j > 0$ for funding variables)
- Elevated volatility risk premia in high-regime periods
- Higher correlations across assets in high-regime periods
- Greater dispersion in external variables during high-regime periods (higher variance ratios)

These expectations are stated ex ante. We will report whether observed patterns conform to expectations, documenting both confirmations and violations.

## 6.3 Crisis Episode Analysis

The third component examines regime behavior during known stress episodes.

### 6.3.1 Episode Selection

Crisis episodes are selected based on external criteria, independent of regime behavior. Selection criteria include:

- Official designations (e.g., NBER recession dates)
- Market-based thresholds (e.g., VIX exceeding historical percentiles)
- Event-based identification (e.g., Lehman bankruptcy, COVID-19 market disruption)

The specific episodes included in the analysis are:

| Episode | Start | Peak | End | Description |
|---------|-------|------|-----|-------------|
| Volmageddon | 2018-02-01 | 2018-02-05 | 2018-02-28 | XIV collapse, VIX spike |
| COVID Crash | 2020-02-20 | 2020-03-16 | 2020-04-30 | Pandemic market crash |
| Rate Shock 2022 | 2022-09-01 | 2022-10-15 | 2022-11-30 | Fed tightening, bond selloff |
| SVB Crisis | 2023-03-08 | 2023-03-13 | 2023-03-31 | Regional banking stress |

Episodes are identified before examining regime behavior during those periods.

### 6.3.2 Regime Dynamics During Episodes

For each episode, we examine:

**Timing.** When does the regime first signal elevated stress relative to the episode onset? Lead or lag is computed relative to the official or market-based start date.

**Magnitude.** How high does the regime indicator rise during the episode? We compare peak regime values across episodes.

**Duration.** How long does the regime remain elevated after episode resolution? Persistence may differ across episode types.

**Recovery.** What is the trajectory of regime decline following episode peaks?

### 6.3.3 Cross-Episode Consistency

We assess whether regime behavior is consistent across episodes of similar type. If the regime captures genuine stress dynamics, it should respond similarly to comparable events. Inconsistent responses---elevated during one crisis but not another of similar severity---would undermine the interpretation.

Consistency is assessed qualitatively through visual inspection and quantitatively through variance decomposition across episodes.

## 6.4 Robustness Design

The analysis includes pre-specified robustness checks to assess sensitivity of conclusions.

### 6.4.1 Threshold Sensitivity

Distributional comparisons are repeated with alternative regime classification thresholds:

- Conservative: $q_t > 0.7$ for high regime
- Liberal: $q_t > 0.3$ for high regime

If conclusions depend critically on the 0.5 threshold, this sensitivity undermines confidence in the findings.

### 6.4.2 Subsample Stability

All analyses are repeated on subsamples:

- Pre-crisis period (before 2007)
- Crisis period (2007-2009)
- Post-crisis period (2010-2019)
- COVID period (2020-present)

If alignment patterns differ substantially across subsamples, this suggests time-varying relationships that complicate interpretation.

### 6.4.3 Observable vs. Latent Regimes

A central discriminant validity test compares the alignment of two regime indicators with external stress proxies: the observable transition function $G_{\text{obs}}$, which conditions on threshold variables such as lagged volatility or the VIX index within the STR-HAR framework, and the latent transition function $G_{\text{ssm}}$, which is inferred from the VRNN state-space model without reference to any observable switching variable. Both indicators are produced by @lim2026latent, but they differ fundamentally in how regime states are identified. Lead-lag analyses, distributional comparisons, and crisis episode trajectories are conducted separately for each indicator.

If latent regimes capture genuine economic structure beyond what observable proxies already reflect, the VRNN-based indicator should exhibit stronger or earlier alignment with held-out stress measures, particularly those not mechanically related to the observable threshold variable. Conversely, if $G_{\text{obs}}$ and $G_{\text{ssm}}$ produce indistinguishable alignment patterns, this would suggest that the latent approach offers no interpretive advantage over simpler observable conditioning. This comparison provides a direct test of whether the additional complexity of latent state inference is justified on economic, rather than purely statistical, grounds.

### 6.4.4 Controlling for Observable Volatility

A key question is whether regimes provide information beyond observable volatility. We repeat distributional comparisons after controlling for realized volatility:

$$\tilde{x}_{j,t} = x_{j,t} - \hat{\mathbb{E}}[x_{j,t} | RV_t]$$

If regime effects disappear after this adjustment, the regime may be proxying for volatility rather than capturing distinct variation.

## 6.5 Multiple Testing Considerations

With multiple external variables and multiple analysis types, false discovery is a concern. We address this through:

**Categorization.** Variables are grouped into categories (funding, risk premia, correlation, events). We first test whether the category as a whole shows alignment, then examine individual variables conditional on category-level significance. Where appropriate, we apply Benjamini-Hochberg false discovery rate control [@benjamini1995controlling].

**Replication logic.** Alignment that appears across multiple variables within a category is more credible than isolated significant correlations. We emphasize patterns rather than individual test statistics.

**Effect size focus.** We report standardized effect sizes alongside p-values. A statistically significant but economically small alignment is less compelling than a large alignment that narrowly misses conventional significance thresholds.

## 6.6 What We Will Report

The results section will report:

1. Cross-correlation functions for each external variable, with confidence bands
2. Tables of conditional means and mean differences across regime states
3. Distributional plots comparing high-regime and low-regime periods
4. Episode-by-episode regime trajectories with event timing markers
5. Robustness tables showing sensitivity to thresholds, subsamples, and controls
6. Summary assessment of alignment patterns across variable categories

The analysis is descriptive. We assess coherence, not causation. Conclusions will be stated conditionally, acknowledging that alignment is consistent with but does not prove economic meaning.
