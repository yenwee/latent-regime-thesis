# 10. Conclusion

This paper examines whether latent volatility regimes, constructed for forecasting purposes, exhibit external economic coherence. The research question is whether regime states align systematically with independent measures of market stress that were not used in their construction. Alignment, if observed, would support the view that regimes capture economically meaningful variation; absence of alignment would suggest regimes are statistically useful but economically opaque.

## 10.1 Summary of Approach

The analysis maintains strict separation between construction and validation. Latent regimes are taken as given from the companion papers [@lim2026latent; @lim2026regime], inferred from realized volatility and returns. Validation variables---funding spreads, volatility risk premia, correlation measures, and crisis indicators---are held out entirely from regime construction. This separation ensures that observed alignment, if any, cannot be mechanical.

The empirical strategy examines lead-lag relationships, distributional differences across regime states, and behavior during known stress episodes. The approach is descriptive: we assess coherence without claiming to identify causal mechanisms. Multiple robustness checks address concerns about threshold sensitivity, subsample stability, and the distinction between regime effects and volatility effects.

## 10.2 Summary of Findings

The analysis reveals systematic alignment between latent regimes and external stress indicators across all four validation dimensions. The regime indicator is contemporaneously correlated with equity volatility measures (VIX $\rho = 0.254$, VVIX $\rho = 0.198$) and exhibits cross-correlation leads with credit spreads of 10 to 12 trading days, though this lead is mediated through observable volatility and does not survive as incremental predictive content once realized volatility is controlled for. Granger causality tests are significant for five of seven external variables. Distributional separation across regime states is substantial, with Cohen's $d$ ranging from 0.52 (TED spread) to 1.47 (VIX), and all Kolmogorov-Smirnov tests rejecting distributional equality at $p < 0.001$. During four pre-specified crisis episodes, the regime activated before the volatility peak in every case, with lead times ranging from 1 day (Volmageddon, SVB) to 30 days (rate shock). The latent indicator normalized faster than its observable counterpart across all episodes, spending 2 to 11 fewer days in the elevated state. Where the regime demonstrates genuine incremental value, it is in anticipating yield curve dynamics and shifts in the volatility risk premium: predictive regressions show that the regime predicts term spread compression at all horizons ($p < 0.013$) even after controlling for realized volatility, and Granger causality from the regime to VVIX is significant at all tested lag orders. Panel analysis across 20 assets confirms that this predictive content is not idiosyncratic. Compared to the observable regime indicator, the latent regime achieves larger distributional effect sizes for volatility-related measures (VVIX: $d = 0.98$ vs. $0.66$) while producing fewer false positive stress signals during crisis episodes.

## 10.3 Implications

### 10.3.1 For Research

The validation framework employed here applies beyond volatility regimes. Any latent variable constructed from one data source can be validated against held-out variables from other sources. The key requirements are strict separation between construction and validation data, pre-specification of analyses, and conservative interpretation of results.

The findings speak to the broader question of what latent variables in finance represent. Forecasting performance establishes statistical utility; external coherence provides evidence about economic content. Both are valuable; neither alone suffices for comprehensive understanding.

### 10.3.2 For Practice

Conditional on alignment, regime indicators may complement existing risk monitoring tools. They summarize volatility dynamics in a form that relates to broader market conditions. This complements rather than replaces observable measures such as the VIX, which capture different aspects of market information.

Caution is warranted in application. Regimes are estimates, not observations; filtered states, not forecasts. Their utility depends on the stability of patterns observed in historical data, which may or may not persist in future conditions.

## 10.4 Limitations Restated

The analysis cannot establish causation, prove meaning, or ensure generalization. Alignment is consistent with economic content but does not demonstrate it conclusively. The evidence supports but does not prove the interpretation that regimes capture genuine market structure. Alternative explanations remain possible.

## 10.5 Concluding Remark

Latent variable models occupy an uncertain position between pure statistical devices and representations of economic phenomena. This paper examines one such model---the volatility regime---and asks whether it falls closer to one pole or the other. The evidence, as presented, informs this question without fully resolving it. Complete resolution may be beyond the reach of observational analysis. What validation provides is a disciplined assessment of coherence, contributing to judgment about interpretation while acknowledging the limits of what data can establish.
