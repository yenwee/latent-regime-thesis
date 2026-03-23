# 2. Conceptual Framing: Validating Latent Variables

Latent variables occupy a peculiar position in empirical research. They are constructed to explain observed data, yet they are not themselves observed. A latent regime state is inferred from volatility dynamics, not measured directly. This indirectness creates both flexibility and risk: flexibility because the latent variable can capture structure that observable proxies miss, risk because the variable may reflect estimation artifacts rather than genuine phenomena.

## 2.1 What Latent Variables Are

A latent variable is an unobserved quantity posited to underlie observed data. In state-space models, the latent state evolves according to specified dynamics and generates observations through a measurement equation. The state is recovered through filtering or smoothing, conditional on the model specification. The recovered state is not "true" in any absolute sense; it is the best estimate given the model and data.

In our context, the latent regime is a low-dimensional state inferred from realized volatility observations. The inference procedure---a Variational Recurrent Neural Network [@chung2015recurrent]---approximates the posterior distribution of latent states given the observed volatility sequence. The resulting regime indicator is a probabilistic object, not a deterministic classification. High values indicate elevated probability of a "high volatility" state, but the state itself is a modeling construct.

## 2.2 Why Validation Is Necessary

The forecasting success documented in the companion papers [@lim2026latent; @lim2026regime] does not, by itself, validate the latent regime as economically meaningful. A model can improve predictions through mechanisms unrelated to economic fundamentals: flexible functional forms, regularization properties, or capacity to capture high-frequency noise. The latent state might be statistically useful without corresponding to any interpretable market condition.

This concern is not merely philosophical. Latent variable models have been criticized for producing "black box" predictions that lack scientific content. If the regime state cannot be related to external phenomena, its value is limited to narrow forecasting applications. It cannot inform economic understanding, policy analysis, or risk management beyond the specific prediction task for which it was optimized.

Validation addresses this concern by examining whether the latent variable exhibits relationships with external data that were not used in its construction. If the regime state, built from volatility data alone, systematically aligns with funding conditions, risk premia, or crisis indicators, this provides evidence that the latent variable captures economically relevant structure. The alignment cannot be mechanical---the external variables were held out---so any correspondence must reflect genuine underlying relationships.

## 2.3 Structural Interpretation vs. External Coherence

Two distinct approaches exist for interpreting latent variables. Structural interpretation attempts to identify what the latent variable "is"---to map it onto a specific economic concept such as investor fear, funding liquidity, or systemic risk. This approach requires strong assumptions and often involves post-hoc narrative construction.

External coherence, by contrast, examines whether the latent variable behaves consistently with multiple economic indicators without claiming identity with any single concept. The regime state may align with funding stress, volatility risk premia, and correlation dynamics simultaneously, without being reducible to any one of these. Coherence is a weaker but more defensible claim than structural interpretation.

This paper adopts the external coherence approach. We assess whether the latent regime exhibits systematic relationships with held-out variables. We do not claim to identify what the regime "represents" in any singular sense. Multiple economic phenomena may contribute to regime variation, and disentangling their relative contributions is beyond the scope of validation analysis.

## 2.4 Construct Validity

The concept of construct validity, drawn from psychometrics [@cronbach1955construct] and extended to econometrics, provides a framework for our analysis. A construct is valid to the extent that it measures what it purports to measure. For latent variables, this requires examining whether the inferred construct relates appropriately to external criteria.

Convergent validity asks whether the latent variable correlates with measures it should theoretically relate to [@campbell1959convergent]. If the regime state captures market stress, it should align with stress indicators. Discriminant validity asks whether the latent variable is distinct from measures it should differ from [@campbell1959convergent]. The regime state should not simply replicate observable volatility; it should contain information beyond what observables provide.

Our validation design incorporates both aspects. We examine correlations with theoretically relevant stress indicators (convergent validity) while testing whether the regime provides incremental information beyond observable measures (discriminant validity).

## 2.5 Alignment Does Not Imply Causation

A critical caveat applies throughout this analysis. Alignment between the latent regime and external indicators does not establish causation in either direction. The regime may lead stress indicators, but this does not mean the regime causes stress. Both may respond to a common underlying factor with different timing. Alternatively, the regime may lag indicators due to estimation smoothing, without implying that indicators drive regimes.

We examine temporal relationships to characterize alignment, not to identify causal mechanisms. Lead-lag patterns are informative about the regime's potential utility for monitoring or anticipation, but they do not support claims about economic transmission channels. This limitation is inherent to validation analysis and is not resolved by more sophisticated statistical methods.

The appropriate conclusion from alignment evidence is conditional: "If the latent regime captures economically meaningful variation, these patterns are consistent with that interpretation." The evidence supports but does not prove the interpretation. Alternative explanations---spurious correlation, common confounders, estimation artifacts---remain possible and cannot be definitively ruled out.
