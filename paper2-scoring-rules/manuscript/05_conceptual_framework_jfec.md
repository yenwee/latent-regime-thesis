# 3. Conceptual Framework

This section develops the conceptual motivation for shared latent regimes. We first establish why volatility regimes may have a systemic component, then explain why observable proxies for this component are inadequate, and finally articulate why partial pooling---shared regime identification with heterogeneous responses---is the appropriate modeling strategy.

## 3.1 Why Regimes May Be Systemic

Volatility regimes reflect the state of market uncertainty. A "high volatility regime" is not simply a period of elevated variance; it is a distinct market state with different dynamics, persistence, and risk characteristics. The question is whether this state is determined independently for each asset or contains a common component across assets.

Several mechanisms suggest a systemic component. First, macroeconomic uncertainty affects multiple asset classes simultaneously. Monetary policy uncertainty, growth shocks, and geopolitical events do not target individual assets; they shift the uncertainty environment for markets broadly. Second, funding and liquidity conditions are market-wide. When funding liquidity tightens, it affects the capacity to take risk across assets, not just in specific markets. Third, investor risk appetite exhibits common variation. Risk-on/risk-off dynamics---the tendency for investors to simultaneously increase or decrease exposure across risky assets---imply correlated regime states.

These mechanisms suggest that while volatility levels may differ across assets, the timing of regime transitions may be synchronized. An equity index and a commodity may have different volatility magnitudes in both calm and turbulent states, but they may transition between states at similar times in response to common drivers.

## 3.2 Why Observable Proxies Fail

If regime states have a systemic component, an obvious approach is to condition regime transitions on observable proxies for market stress: the VIX, credit spreads, or funding liquidity measures. This is the approach taken by threshold and smooth transition models that use observables as transition variables.

Observable proxies have two limitations. First, they are noisy measures of the latent regime state. The VIX reflects S&P 500 option-implied volatility and market positioning, not a pure measure of regime. It can spike due to positioning dynamics even when the underlying regime state has not changed. Credit spreads embed credit risk premia that may move independently of volatility regimes. Any observable proxy conflates the regime signal with measurement noise.

Second, observable proxies may lag the true regime transition. Implied volatility responds to realized volatility with a delay. Credit spreads adjust as defaults materialize. If the latent regime state shifts before these observables update, conditioning on observables will identify regime transitions too late for optimal forecasting.

A latent state inferred from realized volatility data sidesteps both problems. It is estimated to maximize forecasting performance rather than derived from an imperfect proxy. And it can, in principle, identify regime transitions as they occur rather than waiting for observable indicators to update.

## 3.3 Why Partial Pooling Is Appropriate

Given that regimes may be systemic, how should we model cross-asset structure? Two extreme approaches bracket the possibilities. Full pooling treats all assets as having identical regime dynamics: a single latent state governs all assets, and all assets respond identically to this state. No pooling treats each asset independently: each has its own latent regime with no cross-asset information.

Neither extreme is appropriate. Full pooling ignores the substantial heterogeneity in how assets respond to market stress. During the 2020 COVID-19 episode, equity volatility spiked dramatically while Treasury volatility remained elevated but less extreme. Gold exhibited safe-haven dynamics distinct from risk assets. A single regime state with identical coefficients cannot capture these differences.

No pooling ignores the cross-asset information that could improve regime identification. When most assets show signs of transitioning to a high-volatility state, this provides evidence about the regime state that is useful for assets where the signal is ambiguous. An asset-specific model must identify regime transitions from a single volatility series; a shared model can exploit cross-sectional evidence.

Partial pooling occupies the middle ground. A shared latent state, inferred from cross-asset observations, governs the timing of regime transitions. But the mapping from this latent state to volatility dynamics is asset-specific. Each asset has its own STR-HAR coefficients that determine how it responds to the shared regime. This preserves heterogeneity in volatility levels and dynamics while exploiting the common information in regime timing.

The partial pooling interpretation is that regimes have both systemic and idiosyncratic components. The systemic component---when markets broadly shift between calm and turbulent states---is captured by the shared latent state. The idiosyncratic component---how a specific asset's volatility responds to this shift---is captured by asset-specific coefficients.

## 3.4 Regime Transmission vs. Variance Transmission

A final conceptual point distinguishes our framework from spillover models. Variance transmission and regime transmission are different phenomena.

Variance transmission occurs when a shock to asset $i$'s volatility causes a subsequent change in asset $j$'s volatility. The shock propagates through linkages between assets---common exposures, portfolio rebalancing, or information channels. This is the phenomenon measured by Diebold-Yilmaz spillover indices and modeled by multivariate GARCH.

Regime transmission occurs when asset $i$ and asset $j$ transition between volatility states at similar times due to a common underlying driver. There is no causal link from $i$ to $j$; rather, both respond to a shared factor that governs when markets shift from calm to turbulent conditions.

The distinction matters for modeling. Variance transmission operates within a regime: conditional on the current regime state, shocks propagate across assets. Regime transmission determines when the regime state changes. A model of variance transmission does not capture regime dynamics; a model of regime dynamics does not capture shock propagation. Our focus is on the latter.

## 3.5 Summary

The conceptual framework can be summarized as follows. Volatility regimes contain a systemic component driven by macroeconomic uncertainty, funding conditions, and risk appetite. Observable proxies for this component are noisy and potentially lagged. A latent state inferred from cross-asset volatility data can identify regime transitions more effectively than either observable proxies or asset-specific models. Partial pooling---shared regime timing with heterogeneous asset responses---is the appropriate modeling strategy. This regime structure is distinct from variance transmission and requires different modeling tools.
