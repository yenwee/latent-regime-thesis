---
title: "Latent State-Space Smooth Transition Models for Realized Volatility Forecasting"
author:
  - "Yen Wee Lim, University of Malaya, Kuala Lumpur, Malaysia"
date: "March 2026"
abstract: |
  We propose Deep-LSTR, integrating Variational Recurrent Neural Networks with Smooth Transition HAR models using inferred latent states as transition variables. Evaluating 21 assets across equity, fixed income, currency, and commodity markets (2015--2025), we demonstrate that latent regime identification yields consistent improvements over observable-transition specifications, winning on 60--80% of individual assets and achieving Model Confidence Set inclusion rates of 89--100% across forecast horizons. The latent transition variable exhibits a regime smoothing mechanism, reaching extreme values on only 19% of days compared to 34% for observable proxies, consistent with fewer false positive regime switches. We apply the Fissler-Ziegel joint elicitability framework for Expected Shortfall evaluation, achieving 1.5% lower FZ0 loss than the HAR benchmark while maintaining proper VaR calibration. Robustness checks confirm insensitivity to latent dimensionality, transition function specification, subsample period, and volatility estimator choice.
keywords: "Realized Volatility, HAR Model, Smooth Transition Regression, Variational State-Space Models, Expected Shortfall, Joint Elicitability"
jel: "C22, C45, C53, G17"
---
