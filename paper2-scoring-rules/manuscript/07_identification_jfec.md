# 5. Identification and Interpretation

This section addresses the interpretation of the shared latent state. We clarify what the latent regime is and is not, how heterogeneity is preserved in the partial pooling framework, and what empirical evidence would be inconsistent with our interpretation.

## 5.1 What the Latent State Is Not

The shared latent state $\mathbf{z}_t$ should not be interpreted as:

**Average volatility.** The latent state is not simply a cross-sectional average of realized volatility. While there is a correlation---the latent state tends to be elevated when average volatility is high---the relationship is not deterministic. The encoder learns to identify regime-relevant features of the cross-asset volatility profile, which may include dispersion, relative rankings, or nonlinear combinations that differ from simple averaging.

**A factor loading.** The latent state does not play the role of a common factor in a factor model. Factor models explain cross-sectional variation: a high factor realization implies high volatility across assets, with magnitude determined by loadings. Our latent state governs regime transitions: a high realization implies that assets are in (or entering) a high-volatility regime, but the volatility level in that regime is determined by asset-specific HAR dynamics.

**An observable proxy.** The latent state is not a smoothed or processed version of an observable like the VIX. It is learned from the realized volatility data to optimize reconstruction and forecasting. It may correlate with the VIX, but it captures information beyond what the VIX provides.

**A causal driver.** We do not claim that the latent state causes volatility dynamics. It is an inferential construct that summarizes the regime state implied by observed volatility. The relationship is closer to filtering than to structural causation.

## 5.2 What the Latent State Is

The shared latent state is best understood as a sufficient statistic for regime timing. Conditional on the latent state, the past volatility history provides no additional information about whether assets are in a high or low volatility regime. The latent state compresses the cross-asset volatility profile into the information relevant for regime classification.

More precisely, the latent state captures the common component of regime variation. Individual assets have both systemic and idiosyncratic regime dynamics. The systemic component---variation in regime state that is shared across assets---is captured by the shared latent state. The idiosyncratic component---asset-specific deviations from the common regime---is captured by the asset-specific transition parameters.

This interpretation is analogous to random effects in panel data. The shared latent state is like a time-varying intercept common to all assets. The asset-specific parameters are like individual fixed effects that modify the response to this common component. Partial pooling allows both sources of variation to be represented.

## 5.3 Heterogeneity in Regime Response

The partial pooling framework preserves heterogeneity through the asset-specific STR-HAR parameters. Consider two assets with different regime sensitivities:

**Asset A: High systemic sensitivity.** This asset has a high $\gamma_i$ (sharp transition) and a low $c_i$ (early transition). It responds quickly and strongly to the shared regime state. When the latent state rises, this asset transitions to the high-volatility regime early and completely.

**Asset B: Low systemic sensitivity.** This asset has a low $\gamma_i$ (gradual transition) and a high $c_i$ (late transition). It responds slowly and partially to the shared regime state. When the latent state rises modestly, this asset remains primarily in the low-volatility regime; only an extreme latent state triggers full transition.

Both assets are governed by the same shared latent state, but they respond differently. Asset A is dominated by the systemic regime component; Asset B is dominated by idiosyncratic dynamics with only modest systemic influence.

This heterogeneity is empirically testable. If partial pooling is appropriate, we should observe: (1) assets differ in their estimated $\gamma_i$ and $c_i$; (2) assets with high estimated systemic sensitivity show larger forecast improvements from shared regimes relative to asset-specific regimes; and (3) assets with low estimated systemic sensitivity show similar or worse performance from shared regimes.

## 5.4 Identification Conditions

For the latent state to be identified, the data must contain information about regime timing that is not fully captured by observable proxies or asset-specific histories. We rely on the following identifying assumptions:

**Cross-asset variation.** There is cross-sectional variation in volatility that contains information about regime state beyond what is available in any single asset. If one asset shows ambiguous signals (moderate volatility increase), other assets can provide disambiguating evidence.

**Regime structure.** Volatility dynamics differ across regimes in ways that are consistent across assets. If each asset had completely idiosyncratic regime dynamics, pooling would not improve identification.

**Observable incompleteness.** Observable proxies like the VIX do not fully capture the regime state. If the VIX were a sufficient statistic for regime timing, the latent state would reduce to a smoothed version of the VIX with no additional information.

These conditions cannot be directly tested but can be indirectly evaluated through forecast performance. If the latent state improves forecasting beyond asset-specific models and observable proxies, this provides evidence that the identifying conditions hold.

## 5.5 Falsification

Several empirical patterns would be inconsistent with our interpretation:

**No improvement from pooling.** If shared latent regimes do not improve forecasting relative to asset-specific regimes for any asset, this suggests that regime dynamics are purely idiosyncratic and no systemic component exists.

**Uniform improvement.** If shared regimes improve forecasting equally for all assets, this suggests that our partial pooling framework underestimates the systemic component. A full pooling model might be more appropriate.

**Observable sufficiency.** If a simple observable proxy (VIX, cross-asset average) achieves the same forecasting performance as the latent state, this suggests that the latent modeling is unnecessary and a threshold model on observables suffices.

**Regime timing divergence.** If the inferred regime states for different asset classes (e.g., equities vs. commodities) diverge substantially, this suggests that "shared" regimes are not truly shared and class-specific models might be preferable.

We evaluate these possibilities in the empirical analysis.

## 5.6 Interpretation Summary

The shared latent state is a compressed representation of cross-asset regime timing. It captures the systemic component of regime variation while allowing asset-specific responses. It is not a volatility factor, an observable proxy, or a causal driver. Its value is demonstrated by forecasting improvement, not by structural interpretation. The partial pooling framework nests both full pooling and no pooling as special cases, allowing the data to determine the appropriate degree of cross-asset information sharing.
