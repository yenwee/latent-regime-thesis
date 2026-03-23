# 2. Related Work

This section positions our contribution relative to three strands of literature: regime modeling in volatility, multivariate volatility and spillovers, and latent factor approaches. We focus on how each strand treats cross-asset regime structure rather than providing a comprehensive survey.

## 2.1 Regime Modeling in Volatility

The volatility regime literature begins with @hamilton1989new's Markov-switching framework, extended to conditional heteroskedasticity by @hamilton1994autoregressive and @gray1996modeling. @haas2004new develop a tractable Markov-switching GARCH specification that allows regime-dependent variance dynamics. @klaassen2002improving shows that regime-switching improves GARCH forecasts, particularly during volatile periods.

Smooth transition models offer an alternative to discrete regime switching. @granger1993modelling and @terasvirta1994specification develop smooth transition autoregressive (STAR) specifications where transitions between regimes are governed by a continuous function of an observable variable. Applied to realized volatility, the STR-HAR model conditions regime transitions on lagged volatility or other observables.

A limitation of both approaches is their treatment of regimes as asset-specific. Each asset has its own regime indicator---whether Markov-switching probabilities or smooth transition functions---estimated from its own volatility history. Cross-asset information is ignored in regime identification. Our framework departs from this tradition by inferring a shared regime state from multivariate observations.

## 2.2 Multivariate Volatility and Spillovers

Multivariate GARCH models [@bollerslev1988capital; @engle2002dynamic] capture cross-asset volatility dynamics through conditional covariance matrices. The BEKK specification allows volatility shocks to transmit across assets, while DCC models separate correlation dynamics from univariate volatility processes. These models capture variance transmission but do not distinguish between shock propagation within a regime and transitions between regimes.

The spillover framework of @diebold2009measuring and @diebold2012better measures directional volatility transmission using forecast error variance decompositions. This approach has been extended to time-varying settings and applied widely to study financial contagion. Network models of volatility [@demirer2018estimating] build on this foundation to characterize the topology of volatility transmission.

These frameworks address a different question than ours. They measure how a shock to asset $i$'s volatility affects asset $j$'s volatility---variance transmission within a given regime state. We are concerned with whether the transition between regimes is synchronized across assets---regime transmission. A market-wide shift from calm to turbulent conditions affects all assets simultaneously but is not captured by variance decompositions, which measure shock propagation rather than state transitions.

## 2.3 Latent Factor and State-Space Approaches

Factor models extract common components from cross-sectional data. @stock2002forecasting show that principal components improve forecasting in large panels. Applied to volatility, latent factor models [@ludvigson2009macro; @jurado2015measuring] identify common volatility factors that explain cross-sectional variation in volatility levels.

State-space models provide a general framework for latent variable inference. @rangapuram2018deep extend this to deep learning settings with recurrent architectures. @krishnan2017structured develop structured inference networks that improve latent state estimation in nonlinear dynamics.

Our approach differs from factor models in its objective. Factor models extract latent components that explain variation in volatility levels---a cross-sectional dimension reduction. Our latent state governs regime transitions---the timing of shifts between volatility states. A high realization of our latent state does not imply high volatility levels across assets; it implies that assets are in (or transitioning to) a high-volatility regime. The distinction is between a level factor and a regime indicator.

## 2.4 This Paper

We combine elements from these literatures while addressing their limitations. Like regime models, we allow volatility dynamics to differ across states. Like multivariate models, we exploit cross-asset information. Like factor models, we extract a common latent component. But we apply the factor structure to regime identification rather than volatility levels, and we model regime transmission rather than variance transmission.

The most closely related work is our companion paper on single-asset latent regime models, which develops the VRNN-based regime identification approach for individual assets. This paper extends that framework to multivariate observations, introducing shared regimes with partial pooling. The question we address is whether the regime structure identified in that work is purely asset-specific or contains a systemic component that can be exploited for improved forecasting.
