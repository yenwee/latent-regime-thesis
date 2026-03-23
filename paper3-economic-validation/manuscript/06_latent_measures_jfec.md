# 4. Latent Regime Measures Used

This section describes the latent regime indicators that are the objects of validation. Two regime indicators are examined from the first companion paper [@lim2026latent]: the latent transition function $G_{\text{ssm}}$ inferred by the VRNN and the observable transition function $G_{\text{obs}}$ based on threshold variables. Both are taken as given; this paper does not re-estimate or modify them.

## 4.1 Asset-Specific Latent Regimes

The first companion paper [@lim2026latent] develops a methodology for inferring latent regime states from individual asset volatility data. For each asset in the panel, a Variational Recurrent Neural Network [@chung2015recurrent] processes the realized volatility sequence and produces a posterior distribution over latent states at each time point.

The latent state is low-dimensional, with two dimensions in the baseline specification [robustness checks in @lim2026latent confirm this choice is optimal]. This state is projected to a scalar regime indicator that governs transitions in a Smooth Transition HAR model [@granger1993modelling; @terasvirta1994specification]. High values of the regime indicator correspond to "high volatility" regime dynamics; low values correspond to "low volatility" dynamics. The transition between regimes is smooth, governed by a logistic function.

The key property of these regimes is that they are inferred without reference to external variables. The VRNN observes realized volatility and, in some specifications, returns. It does not observe funding conditions, option-implied measures, or macroeconomic indicators. Whatever structure the regime captures is extracted from the volatility data itself.

For validation purposes, we use the posterior mean of the regime indicator at each time point. This provides a continuous measure ranging from zero to one, interpretable as the probability of being in the high-volatility regime. Thresholding this measure at 0.5 produces a binary regime classification when discrete states are required.

## 4.2 Observable Regime Indicator

@lim2026latent also produces an observable regime indicator, $G_{\text{obs}}$, which conditions on threshold variables such as lagged realized volatility or the VIX index within the STR-HAR framework. Unlike the latent indicator, $G_{\text{obs}}$ does not involve neural network inference; it applies a logistic transition function to an observable conditioning variable.

The observable indicator serves as a discriminant validity benchmark. If the latent regime captures genuine economic structure beyond what observable proxies already reflect, then $G_{\text{ssm}}$ should exhibit stronger or earlier alignment with held-out stress measures than $G_{\text{obs}}$. Conversely, if both indicators produce indistinguishable alignment patterns, this would suggest that the additional complexity of latent state inference is not justified on economic grounds.

Both indicators range from zero to one and are available at daily frequency for each asset in the panel. Their distributional properties differ substantially: $G_{\text{ssm}}$ is concentrated near 0.5 (standard deviation approximately 0.11), while $G_{\text{obs}}$ has wider spread (standard deviation approximately 0.27). This difference reflects the latent indicator's more conservative regime classification.

## 4.3 Construction Without External Variables

A critical feature of both regime types is their construction without external validation variables. The regimes are optimized for volatility forecasting, not for alignment with stress indicators. The VRNN loss function penalizes reconstruction error and forecasting loss; it does not reward correlation with funding measures or risk premia.

This separation is essential for valid inference. If the regime were constructed to maximize correlation with stress indicators, any observed alignment would be mechanical rather than informative. The regime would "align" with stress by construction, providing no evidence of genuine economic content.

Because the regimes are constructed from volatility data alone, any alignment with held-out variables must arise from underlying relationships in the data. The volatility dynamics that the regime captures must themselves be related to funding conditions, risk premia, or other stress measures. This is the substantive hypothesis that validation examines.

## 4.4 What the Regimes Are Not

To clarify interpretation, we note what the latent regimes are not:

**Not direct measures of volatility.** The regime state is correlated with volatility levels but is not identical to them. The regime may be elevated while volatility is moderate, or low while volatility is high, if the regime captures persistence or transition dynamics rather than levels.

**Not observable indicators.** The regime is not a transformation of VIX, realized volatility, or any observable measure. It is learned to optimize forecasting, which may produce a variable distinct from any single observable.

**Not structural parameters.** The regime is not a parameter of an economic model. It is a statistical construct that may or may not correspond to theoretical concepts like "fear" or "uncertainty."

**Not predictions.** The regime state is a filtered estimate of the current latent state, not a forecast of future volatility. It summarizes information about the current regime, not future outcomes.

These distinctions matter for interpreting validation results. Alignment between regimes and stress indicators is meaningful precisely because regimes are none of these things. If regimes were simply transformed volatility, alignment would be mechanical; if they were observable indicators, they would provide no new information.
