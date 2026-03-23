# Abstract

We propose Deep-LSTR, integrating Variational Recurrent Neural Networks with Smooth Transition HAR models using inferred latent states as transition variables. Evaluating 21 assets across equity, fixed income, currency, and commodity markets (2015--2025), we demonstrate that latent regime identification yields lower median QLIKE loss than HAR benchmarks, winning on 60--80% of individual assets depending on forecast horizon and achieving Model Confidence Set inclusion rates of 89--100% across horizons. The improvement is consistent with a regime smoothing mechanism: the latent transition variable reaches extreme values on only 19.0% of days versus 34.0% for observable proxies, filtering transient volatility noise. Crucially, we apply Fissler-Ziegel joint elicitability for Expected Shortfall evaluation, addressing the methodological gap whereby most deep-learning volatility studies employ invalid ES scoring rules.

---

**Keywords**: Realized Volatility, HAR Model, Smooth Transition Regression, Variational State-Space Models, Expected Shortfall, Joint Elicitability

**JEL Classification**: C22, C45, C53, C58, G17
