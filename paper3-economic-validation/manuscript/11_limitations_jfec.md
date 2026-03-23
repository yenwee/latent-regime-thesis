# 9. Limitations

This section documents the limitations of the validation analysis. Transparent acknowledgment of limitations is preferable to allowing readers to discover them independently. Some limitations are inherent to validation exercises; others are specific to our implementation.

## 9.1 Fundamental Limitations

### 9.1.1 Correlation Is Not Causation

This limitation pervades the analysis. Alignment between regimes and external indicators does not establish that regimes cause, predict, or represent those indicators. Both may respond to unobserved common factors. The temporal precedence we examine (lead-lag structure) does not identify causation; leads may reflect faster response to common shocks rather than causal influence.

We cannot resolve this limitation within the validation framework. Causal identification would require experimental variation or quasi-experimental designs that are not available for aggregate market conditions. The limitation is inherent, not correctable.

### 9.1.2 Validation Cannot Prove Meaning

Even strong alignment does not prove that regimes are "meaningful" in any absolute sense. Meaning is a claim about interpretation, not a property that data can confirm. A variable can exhibit every pattern we test for and still fail to correspond to the economic concept we imagine. The gap between statistical patterns and conceptual interpretation cannot be closed by analysis.

What validation can do is assess consistency. Consistent alignment with multiple indicators across multiple episodes is more compatible with meaningful interpretation than inconsistent patterns. But consistency supports rather than establishes interpretation.

### 9.1.3 The Validation Variables Are Themselves Imperfect

The external indicators used for validation are proxies for economic concepts, not direct measurements. The TED spread proxies for funding stress but may reflect other factors [@schwarz2019mind]. The VIX proxies for expected volatility but embeds risk premia [@bekaert2014vix]. Implied correlation indices depend on option market liquidity and pricing models.

Alignment with imperfect proxies establishes alignment with those proxies, not with the underlying concepts they represent. If funding stress is measured with error, alignment with measured funding stress understates (or, potentially, overstates) alignment with true funding stress.

## 9.2 Methodological Limitations

### 9.2.1 Pre-Specification Is Incomplete

Despite efforts to pre-specify analyses, researcher degrees of freedom remain. The selection of external variables, the choice of lead-lag windows, the threshold for regime classification---all involve judgment. Post-hoc rationalization of choices that yield favorable results is a risk even with documented pre-specification.

We mitigate this concern through comprehensive reporting of robustness checks and transparent documentation of analytical choices. But complete elimination of researcher discretion is impossible.

### 9.2.2 Multiple Testing Without Full Correction

The analysis examines multiple external variables, multiple leads and lags, and multiple robustness specifications. Full correction for multiple testing would substantially reduce power, potentially obscuring genuine effects. Our strategy---emphasizing patterns across categories rather than individual test statistics---addresses this concern imperfectly.

Some nominally significant results may reflect chance. Patterns that appear robust across categories are more credible, but the distinction between genuine and spurious alignment cannot be drawn with certainty.

### 9.2.3 Sample Period Dependence

The validation covers a specific historical period with particular characteristics: the global financial crisis, post-crisis regulatory changes, the COVID-19 shock. Alignment patterns observed in this sample may not generalize to future periods with different characteristics.

This concern is particularly acute given structural changes in market microstructure, central bank policy, and volatility products. The relevance of historical patterns to future conditions is uncertain.

## 9.3 Data Limitations

### 9.3.1 External Variable Availability

Not all desired external variables are available at the required frequency or for the full sample period. Some measures are proprietary or available only from commercial data vendors. The validation necessarily uses available rather than ideal variables.

Missing variables may be more informative than available ones. Alignment with available variables does not preclude misalignment with unavailable measures that better capture the relevant economic conditions.

### 9.3.2 Measurement Frequency Mismatch

Latent regimes are constructed from daily realized volatility (computed from intraday data). Some external variables are available only at daily close or lower frequency. This mismatch may introduce measurement error that obscures genuine relationships.

Higher-frequency external measures would permit cleaner analysis but are often unavailable or unreliable. The daily frequency represents a practical compromise.

### 9.3.3 Survivorship and Selection in Asset Panel

The asset panel used for regime construction exhibits survivorship bias: assets must have sufficient history and liquidity to be included. This selection may affect the generality of regime patterns. Regimes inferred from a broader panel might exhibit different characteristics.

## 9.4 Interpretation Limitations

### 9.4.1 Regimes Are Low-Dimensional

The latent regime is a low-dimensional summary of a high-dimensional phenomenon. Market stress has many facets---funding, credit, liquidity, volatility, correlation---and the regime cannot capture all of them. Alignment with some dimensions may coexist with misalignment on others.

This limitation is inherent to low-dimensional latent variable models. Higher-dimensional specifications might capture additional facets but would complicate interpretation and reduce parsimony.

### 9.4.2 Regime Boundaries Are Arbitrary

The distinction between "high regime" and "low regime" depends on a threshold applied to a continuous indicator. The 0.5 threshold has an interpretation (equal posterior probability) but remains a choice. Different thresholds yield different classifications and potentially different alignment patterns.

We address this through threshold sensitivity analysis, but the arbitrariness cannot be eliminated. Continuous regime indicators avoid this limitation at the cost of complicating distributional comparisons.

### 9.4.3 Case Study Generalization Is Limited

The case study episodes, however carefully selected, represent a small and potentially unrepresentative sample of stress events. Patterns observed across four episodes may not generalize to future crises of different character.

Case studies illustrate rather than prove. Their limitations are acknowledged in design; they should not be weighted beyond their evidentiary role.

## 9.5 What Would Change Our Conclusions

Transparency requires stating conditions under which we would revise conclusions:

- **Strong alignment reversed by controls**: If regime effects disappear after controlling for observable volatility, the regime may be proxying for volatility rather than capturing distinct variation. This would substantially weaken the economic interpretation.

- **Inconsistency across episodes**: If regime behavior differs qualitatively across crises of similar severity, the interpretation of regimes as capturing stable stress dynamics would be undermined.

- **Category-specific alignment only**: If alignment appears only within a single category (e.g., funding variables but not risk premia), the claim of broad economic coherence would be weakened.

- **Subsample instability**: If alignment patterns differ substantially across subsamples, the regime's economic content may be time-varying or spurious.

These conditions are stated ex ante. We will report whether they hold and adjust interpretations accordingly.

## 9.6 Limitations We Accept

Some limitations we accept as inherent to the research design:

- We cannot prove causation; we assess coherence.
- We cannot establish meaning; we test consistency.
- We cannot eliminate researcher discretion; we document choices.
- We cannot ensure generalization; we report the sample.

These limitations are not flaws to be corrected but boundaries to be acknowledged. Validation analysis within these boundaries provides useful evidence; claims beyond these boundaries are unsupported.
