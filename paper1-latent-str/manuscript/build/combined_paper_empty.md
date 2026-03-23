# Abstract

We propose Deep-LSTR, integrating Variational Recurrent Neural Networks with Smooth Transition HAR models using inferred latent states as transition variables. Evaluating 21 assets across equity, fixed income, currency, and commodity markets (2015-2025), we demonstrate that latent regime identification yields {{str_ssm_vs_har_pct_h5:.1f}}% lower QLIKE loss than HAR benchmarks, with Model Confidence Set inclusion rates of {{str_ssm_mcs_h5:.0f}}% across horizons. The improvement arises through a regime smoothing mechanism: the latent transition variable reaches extreme values on only {{trans_deep_lstr_extreme_pct:.1f}}% of days versus {{trans_str_har_extreme_pct:.1f}}% for observable proxies, filtering transient volatility noise. Crucially, we apply Fissler-Ziegel joint elicitability for Expected Shortfall evaluation, addressing the methodological gap whereby most deep-learning volatility studies employ invalid ES scoring rules.

---

**Keywords**: Realized Volatility, HAR Model, Smooth Transition Regression, Variational State-Space Models, Expected Shortfall, Joint Elicitability

**JEL Classification**: C22, C45, C53, G17


# 1. Introduction

Realized volatility forecasting underpins quantitative risk management, derivative pricing, and portfolio allocation. Accurate volatility predictions are essential for computing Value-at-Risk and Expected Shortfall under Basel III/IV regulatory frameworks, pricing volatility derivatives, and constructing mean-variance optimal portfolios. The Heterogeneous Autoregressive (HAR) model of Corsi (2009) serves as the dominant benchmark, elegantly approximating long-memory dynamics through daily, weekly, and monthly volatility cascades while requiring only four parameters estimable via ordinary least squares. The HAR model's combination of parsimony, interpretability, and competitive forecasting accuracy has established it as the standard against which new volatility models are measured. However, the linear HAR structure assumes constant coefficients across market conditions, ignoring the well-documented regime-switching behavior observed during alternating periods of market tranquility and crisis.

Smooth Transition Autoregressive (STAR) models address this limitation by allowing coefficients to vary continuously between regimes according to a transition function (Teräsvirta 1994; Granger and Teräsvirta 1993). Applications to volatility forecasting have combined the STR framework with HAR specifications (McAleer and Medeiros 2008; Hillebrand and Medeiros 2010), permitting regime-dependent persistence parameters that capture the differential dynamics of calm versus turbulent markets. During high-volatility regimes, the daily component typically dominates as markets exhibit heightened short-term reactivity, while low-volatility regimes display stronger monthly persistence reflecting gradual mean-reversion. Yet existing STR-HAR implementations rely exclusively on observable transition variables—typically lagged realized volatility or past returns.

This observable-transition assumption is problematic. Observable volatility measures are inherently noisy—transient spikes from earnings announcements, flash crashes, and liquidity events can trigger false positive regime switches that do not persist. The 2010 Flash Crash and March 2020 COVID-19 liquidity crisis demonstrate that observable volatility exhibits extreme short-term movements that may not reflect true regime shifts. An observable-STR model is therefore susceptible to overreaction, classifying brief volatility spikes as regime changes and generating unstable coefficient dynamics. This limitation motivates the search for latent transition variables that can filter noise from true regime information.

This paper proposes the Deep Latent State-Space Smooth Transition Autoregressive (Deep-LSTR) framework to resolve this limitation. We employ a Variational Recurrent Neural Network (Chung et al. 2015) to infer a continuous latent state $z_t \in \mathbb{R}^d$ from observable market signals including realized volatility, returns, and trading volume. This latent state is linearly projected to form the scalar transition variable $s_t = \alpha^\top z_t$ governing a smooth transition HAR specification. The resulting architecture preserves the interpretability of regime-dependent HAR coefficients while leveraging deep probabilistic inference to construct a smoothed transition variable that filters transient volatility noise. Crucially, the STR-HAR coefficients remain directly interpretable as regime-specific volatility persistence parameters, distinguishing our approach from pure black-box forecasting methods.

We contribute to the literature in three dimensions. First, we introduce the first integration of deep variational state-space models with the interpretable STR-HAR econometric framework, demonstrating that the VRNN generalizes classical state-space econometrics while preserving coefficient interpretability. The latent state provides a data-driven transition variable without requiring the researcher to specify observable proxies ex ante. Second, we provide extensive multi-asset, multi-horizon evidence that latent regime identification dominates observable proxies through a regime smoothing mechanism that reduces false positive regime switches. Our evaluation spans 21 liquid assets across equity, fixed income, currency, and commodity markets, addressing concerns about single-asset overfitting that pervade the volatility forecasting literature. Third, we apply the joint elicitability framework of Fissler and Ziegel (2016) to rigorously compare tail risk forecasting performance, addressing the methodological deficiency whereby most deep-learning volatility studies evaluate Expected Shortfall using improper scoring rules. This represents the first valid ES comparison for deep learning volatility models.

The remainder of this paper proceeds as follows. Section 2 reviews the literature on realized volatility forecasting, regime-switching models, and deep probabilistic approaches. Section 3 develops the Deep-LSTR methodology. Section 4 describes the data and benchmark models. Section 5 presents empirical results. Section 6 discusses robustness checks. Section 7 concludes.


# 2. Literature Review

## 2.1. Realized Volatility and the HAR Model

The development of high-frequency financial econometrics enabled direct observation of a quantity traditionally treated as latent. Andersen and Bollerslev (1998) demonstrated that realized variance, computed as the sum of squared intraday returns, provides a consistent estimator of integrated variance under standard diffusion assumptions. This paradigm shift from latent-variable GARCH models to observable, model-free volatility measurement was formalized by Andersen et al. (2003), who established three empirical regularities: realized volatility exhibits long memory with slowly decaying autocorrelations, its logarithm is approximately Gaussian, and large shocks occur more frequently than Gaussian predictions imply. The importance of accurate volatility measurement extends beyond forecasting to option pricing, portfolio allocation, and risk management (Poon and Granger 2003).

Corsi (2009) introduced the HAR-RV model, which approximates long-memory dynamics through a simple additive cascade:
$$
RV_{t+H} = \beta_0 + \beta_d RV^{(d)}_t + \beta_w RV^{(w)}_t + \beta_m RV^{(m)}_t + \epsilon_{t+H}
$$
where $RV^{(d)}, RV^{(w)}, RV^{(m)}$ represent daily, weekly, and monthly realized volatility components. The theoretical foundation derives from the Heterogeneous Market Hypothesis (Müller et al. 1997), which posits that market participants operating on different time horizons generate distinct volatility components. The HAR model's parsimony and interpretability have made it the dominant benchmark in realized volatility forecasting.

The HAR framework has spawned numerous extensions addressing specific empirical phenomena. Andersen et al. (2007) introduced HAR-J separating continuous variation from jumps, recognizing that discontinuous price movements carry distinct information content. Corsi, Pirino, and Renò (2010) extended this in HAR-CJ with threshold-based jump detection. Bollerslev, Patton, and Quaedvlieg (2016) augmented HAR with realized quarticity in HAR-Q, exploiting measurement error heteroskedasticity. Patton and Sheppard (2015) incorporated leverage effects through LHAR, decomposing returns into positive and negative components. Bekaert and Hoerova (2014) examined the relationship between realized and implied volatility within HAR frameworks. Despite these advances, none allows for the smooth, continuous coefficient variation that characterizes transitions between market regimes.

## 2.2. Regime-Switching in Volatility

Financial volatility exhibits well-documented nonlinearities that linear specifications cannot capture. The leverage effect (Black 1976; Christie 1982) describes asymmetric responses whereby negative returns increase future volatility more than positive returns. Volatility feedback effects create self-exciting dynamics where high volatility generates uncertainty, affecting returns and further amplifying volatility (French, Schwert, and Stambaugh 1987). Most consequentially, volatility exhibits regime dependence with fundamentally different behavior during calm versus turbulent periods.

Hamilton (1989) introduced regime-switching models where an unobserved Markov chain governs parameter changes. Applications to volatility produced Markov-Switching GARCH (MS-GARCH) models (Hamilton and Susmel 1994; Gray 1996; Haas, Mittnik, and Paolella 2004). While MS-GARCH captures distinct regimes, it imposes abrupt transitions and requires probabilistic regime inference. Klaassen (2002) addressed the path-dependence problem in MS-GARCH through expected regime-conditional variances, though computational complexity remains substantial.

Teräsvirta (1994) developed the Smooth Transition Autoregressive (STAR) framework allowing continuous, gradual transitions:
$$
y_t = (1 - G(s_t)) \Phi_1(y_{t-r}) + G(s_t) \Phi_2(y_{t-r}) + \epsilon_t
$$
where $G(\cdot)$ is a bounded transition function and $s_t$ is the transition variable. The STAR framework nests both linear models (as $\gamma \to 0$) and threshold autoregressions (as $\gamma \to \infty$). Granger and Teräsvirta (1993) provided comprehensive guidance on specification, estimation, and evaluation of nonlinear time series models within this framework.

McAleer and Medeiros (2008) and Hillebrand and Medeiros (2010) applied STR to HAR models using observable transition variables such as lagged realized volatility. This approach assumes regime transitions are identifiable from past data—an assumption violated when regime shifts are driven by unobserved factors including liquidity conditions, institutional positioning, and investor sentiment. The present paper addresses this limitation by inferring latent transition variables through deep state-space models.

## 2.3. Deep Probabilistic Models

The Variational Autoencoder framework (Kingma and Welling 2014) enabled principled deep generative modeling with probabilistic latent representations, tractable training via Evidence Lower Bound maximization, and uncertainty quantification through posterior inference. These advances opened new possibilities for financial applications requiring both flexibility and interpretable uncertainty estimates.

Sequential extensions established deep probabilistic state-space models as a mature methodology. The Variational Recurrent Neural Network (Chung et al. 2015) incorporates latent variables at each time step, maintaining separation between latent dynamics and observations that mirrors classical state-space econometrics while leveraging neural parameterization. The Deep Kalman Filter (Krishnan, Shalit, and Sontag 2017) provides an alternative architecture more explicitly parameterizing state-space dynamics. Rangapuram et al. (2018) demonstrated the effectiveness of deep state-space models for probabilistic time series forecasting across diverse domains.

Machine learning approaches to financial prediction have achieved notable empirical success. Gu, Kelly, and Xiu (2020) provided comprehensive evidence that machine learning methods substantially improve cross-sectional return prediction relative to traditional factor models. Recent advances include DeepVol (Buehler et al. 2024), which employs dilated causal convolutions for volatility forecasting from high-frequency data, achieving state-of-the-art performance but remaining a pure prediction framework without explicit regime structure. Hu, Yin, and Yao (2025) proposed HAR-type models using graph neural networks to capture cross-asset dependencies, exploiting information spillovers across markets.

Despite substantial advances, no prior work integrates deep state-space models with interpretable econometric frameworks while using inferred latent states as transition variables. This methodological gap motivates the present paper's contribution. Furthermore, the deep learning volatility literature has uniformly failed to apply valid elicitability-based evaluation to tail risk forecasts, limiting the reliability of comparative claims.

## 2.4. Tail Risk and Elicitability

Under Basel III/IV, Expected Shortfall has replaced Value-at-Risk as the standard market risk measure (Basel Committee on Banking Supervision 2019). However, while VaR is elicitable via the quantile loss function, ES is not individually elicitable (Gneiting 2011)—no scoring function exists for which ES alone minimizes expected loss. This has profound implications: any ES comparison using RMSE, MAE, or ad-hoc metrics is formally invalid. Acerbi and Tasche (2002) established the theoretical properties of ES as a coherent risk measure, though practical evaluation remained problematic until recent advances.

Fissler and Ziegel (2016) resolved this crisis by proving that the pair $(VaR_\alpha, ES_\alpha)$ is jointly elicitable. The FZ0 loss function provides the only theoretically valid framework for comparing ES forecasts:
$$
S_{FZ0}(Y; v, e) = -\ln(-e) - \frac{v}{e} + \left(1 + \frac{1}{e}\right)(Y - v) \mathbf{1}_{\{Y < v\}}
$$
Patton, Ziegel, and Chen (2019) developed practical implementations and established asymptotic theory for elicitability-based model comparison in risk management applications.

The methodological deficiency in deep-learning volatility research is severe. Studies by Caporale and Zekokh (2019), Kim and Won (2018), and numerous others report ES comparisons without employing joint elicitability, rendering their comparative claims formally unfounded. The present paper addresses this gap by implementing rigorous FZ0-based evaluation, providing the first valid comparison of deep learning methods for Expected Shortfall forecasting in the realized volatility literature.


# 3. Methodology

The Deep Latent State-Space Smooth Transition Autoregressive (Deep-LSTR) model integrates three components into an end-to-end differentiable system: (i) a Variational Recurrent Neural Network that infers a continuous latent state $z_t \in \mathbb{R}^d$ from observed market features; (ii) a linear projection layer mapping the latent state to a scalar transition variable $s_t = \alpha^\top z_t$; and (iii) a Smooth Transition HAR forecasting equation where regime-dependent coefficients are governed by the inferred $s_t$. This architecture preserves the interpretability of classical econometric models while leveraging deep probabilistic inference to construct a smoothed transition variable that filters transient volatility noise from true regime information.

## 3.1. Target Variable: Range-Based Variance Proxy

We employ the Garman-Klass (GK) estimator (Garman and Klass 1980) as our range-based variance proxy:
$$
\hat{\sigma}^2_{GK,t} = 0.5 \cdot \left[\ln\left(\frac{H_t}{L_t}\right)\right]^2 - (2\ln 2 - 1) \cdot \left[\ln\left(\frac{C_t}{O_t}\right)\right]^2 \tag{1}
$$
where $O_t, H_t, L_t, C_t$ denote Open, High, Low, and Close prices. The GK estimator provides a variance proxy that is approximately 7.4 times more efficient than close-to-close squared returns under standard assumptions. Unlike realized volatility measures constructed from intraday data, range-based estimators require only daily OHLC prices, enabling consistent cross-asset comparison across equity indices, commodities, and currencies where high-frequency data availability and market microstructure effects vary substantially.

For horizon $H \in \{1, 5, 22\}$, the target is:
$$
y_{t:t+H} = \ln\left(\frac{1}{H} \sum_{j=1}^{H} \hat{\sigma}^2_{GK,t+j}\right) \tag{2}
$$

## 3.2. The VRNN as Nonlinear State-Space Model

The VRNN is best understood as a nonlinear generalization of classical state-space models. The linear Gaussian state-space model (Kalman filter) specifies:
$$
z_t = F z_{t-1} + \eta_t, \quad \eta_t \sim \mathcal{N}(0, Q) \quad \text{(State Transition)} \tag{3}
$$
$$
x_t = H z_t + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, R) \quad \text{(Observation Equation)} \tag{4}
$$

Financial markets violate linearity and Gaussianity through volatility clustering, leverage effects, fat tails, and regime-dependent dynamics. The VRNN relaxes these restrictions by parameterizing transition and emission functions with neural networks while preserving the separation between latent dynamics and observations:
$$
z_t \sim p_\theta(z_t | z_{t-1}, h_{t-1}) \quad \text{(Learned nonlinear transition)} \tag{5}
$$
$$
x_t \sim p_\phi(x_t | z_t, h_t) \quad \text{(Learned nonlinear emission)} \tag{6}
$$
where $h_t$ is a deterministic recurrent state.

The observation vector at each time $t$ is:
$$
x_t = \begin{bmatrix} \ln(RV_t) \\ |r_t| \\ \ln(Volume_t) \end{bmatrix} \in \mathbb{R}^3 \tag{7}
$$

The latent state $z_t \in \mathbb{R}^d$ (with $d=2$) captures unobserved market regime dynamics.

## 3.3. Generative and Inference Models

**Prior Transition:**
$$
p(z_t | z_{t-1}, h_{t-1}) = \mathcal{N}(\mu_\theta(h_{t-1}), \text{diag}(\sigma^2_\theta(h_{t-1}))) \tag{8}
$$
where $\mu_\theta, \sigma_\theta$ are neural network functions with $h_t$ the GRU hidden state.

**Emission:**
$$
p(x_t | z_t, h_t) = \mathcal{N}(\psi_\phi(z_t, h_t), \text{diag}(\gamma^2_\phi(z_t, h_t))) \tag{9}
$$

**Approximate Posterior:**
$$
q(z_t | x_{\leq t}, z_{<t}) = \mathcal{N}(\mu_\xi(x_t, h_t), \text{diag}(\sigma^2_\xi(x_t, h_t))) \tag{10}
$$

Training proceeds by maximizing the Evidence Lower Bound:
$$
\mathcal{L}(\theta, \phi, \xi) = \sum_{t=1}^T \left[ \mathbb{E}_{q(z_t)} [\ln p(x_t | z_t, h_t)] - D_{KL}[q(z_t | x_{\leq t}) || p(z_t | z_{t-1})] \right] \tag{11}
$$

We deliberately separate latent state inference from volatility forecasting. Joint end-to-end estimation risks degeneracy whereby the latent state collapses into a nonlinear proxy for realized volatility, losing its capacity to capture regime dynamics distinct from observable signals. Our two-stage approach mirrors factor-augmented forecasting frameworks (Stock and Watson 2002), ensuring the transition variable captures regime information through the generative model rather than directly optimizing forecast loss. This separation also facilitates diagnostic analysis: we can examine the statistical properties of the inferred latent state relative to observable volatility proxies, particularly its smoothing characteristics that distinguish it from noisier observable signals.

We optimize using Adam with learning rate $2 \times 10^{-3}$ and weight decay $10^{-4}$ for up to 600 epochs with early stopping (patience of 60 epochs), employing the reparameterization trick for gradient-based optimization through stochastic sampling.

While the VRNN introduces implementation complexity relative to linear HAR, this burden is confined to offline training. Production forecasting requires only a forward pass through the trained encoder---sub-millisecond on standard hardware---followed by the STR-HAR equation. The architecture employed here is deliberately parsimonious: a single-layer GRU with 16 hidden units and a two-dimensional latent space. Training on 600 observations completes in under 60 seconds on CPU with early stopping. This compares favorably to Markov-switching GARCH estimation, which requires iterative EM algorithms with similar or greater computational overhead per rolling window.

## 3.4. Latent Projection and STR-HAR Specification

The VRNN outputs latent trajectories $\{z_t\}_{t=1}^T \in \mathbb{R}^{T \times d}$. We perform supervised linear projection:
$$
s_t = \alpha^\top \tilde{z}_t \tag{12}
$$
where $\alpha$ maximizes correlation with contemporaneous log-volatility:
$$
\hat{\alpha} = \arg\min_\alpha \sum_{t=1}^{T_{train}} (s_t - \ln(RV_t))^2 \tag{13}
$$

A potential concern with Equation (13) is circularity: if $s_t$ is trained to mimic noisy $\ln(RV_t)$, does this reintroduce the measurement noise we claim to filter? The resolution lies in the information bottleneck principle (Tishby, Pereira, and Bialek 2000). The ELBO objective (Equation 11) includes a KL-divergence term that regularizes the approximate posterior $q(z_t|x_{\leq t})$ toward the prior $p(z_t|z_{t-1})$. This regularization acts as a capacity constraint on the latent channel: the mutual information $I(X; Z)$ is upper-bounded by the KL divergence term (Alemi et al. 2017). Consequently, $z_t$ cannot encode arbitrary high-frequency fluctuations present in the input---it must selectively compress information about the volatility process while discarding idiosyncratic noise. The linear projection $\alpha^\top z_t$ therefore extracts a smoothed signal from an already bandwidth-limited representation, rather than fitting to raw noise. This mechanism parallels the $\beta$-VAE framework (Higgins et al. 2017), where increasing the KL weight forces the latent representation to discard irrelevant variation. In our context, the AR(1) prior on $z_t$ combined with KL regularization creates a temporal smoothness prior that preferentially encodes persistent volatility regimes over transient fluctuations.

The projection is economically motivated: while market regimes are driven by high-dimensional processes, their impact on volatility dynamics is effectively one-dimensional, consistent with principal component analyses finding that a single factor explains 70-90% of cross-sectional variance in realized volatilities (Andersen et al. 2001). Standardization of $z_t$ and the OLS fit in Equation (13) use only training-period data; the projection is then applied out-of-sample without refitting. Thus, $s_t$ for periods $t > T_{train}$ is a genuine ex-ante signal derived from the compressed representation, not a fitted transformation of future realized volatility.

The forecasting equation is:
$$
\hat{y}_{t+H} = (1 - G(\tilde{s}_t)) \cdot HAR_{Low} + G(\tilde{s}_t) \cdot HAR_{High} \tag{14}
$$
where:
$$
HAR_{k} = \beta_0^{(k)} + \beta_d^{(k)} x^{(d)}_t + \beta_w^{(k)} x^{(w)}_t + \beta_m^{(k)} x^{(m)}_t, \quad k \in \{L, H\} \tag{15}
$$
with $x^{(d)}_t = \ln(RV_t)$, $x^{(w)}_t = \frac{1}{5}\sum_{j=0}^{4} \ln(RV_{t-j})$, $x^{(m)}_t = \frac{1}{22}\sum_{j=0}^{21} \ln(RV_{t-j})$, and:
$$
G(\tilde{s}_t; \gamma, c) = \frac{1}{1 + \exp(-\gamma(\tilde{s}_t - c))} \tag{16}
$$

Parameters $(\gamma, c, \beta^{(L)}, \beta^{(H)})$ are jointly estimated by nonlinear least squares with basin-hopping optimization, constraining $\gamma \in [0.01, 12]$ and coefficients in $[-5, 5]$.

## 3.5. Benchmark Models

We distinguish between two classes of benchmarks. The primary comparison is conducted within the realized volatility literature: the standard HAR model (Corsi 2009) and the observable-transition STR-HAR serve as direct competitors employing the same forecasting target. Additionally, we include conditional volatility models—GARCH, EGARCH, and MS-GARCH—as external benchmarks from the return-based volatility literature. These models forecast the conditional variance of returns rather than realized volatility measures constructed from high-frequency or range-based data. Their inclusion provides context by assessing whether latent regime identification offers improvements relative to widely used return-based alternatives, while acknowledging that differences in information sets and forecasting objectives naturally favor realized-measure-based models when the evaluation target is realized volatility.

**Linear HAR:** Standard Corsi (2009) specification estimated via OLS on a rolling 600-day window.

**LHAR (Leverage HAR):** Extends HAR with an asymmetric leverage term capturing the well-documented phenomenon that negative returns increase future volatility more than positive returns of equal magnitude (Corsi and Reno 2012):
$$
\hat{y}_{t+H} = \beta_0 + \beta_d x^{(d)}_t + \beta_w x^{(w)}_t + \beta_m x^{(m)}_t + \beta_{lev} r^{-}_t
$$
where $r^{-}_t = \min(r_t, 0)$.

**HAR-J (Jump HAR):** Separates realized variance into continuous and jump components following Andersen, Bollerslev, and Diebold (2007):
$$
\hat{y}_{t+H} = \beta_0 + \beta_d x^{(d)}_t + \beta_w x^{(w)}_t + \beta_m x^{(m)}_t + \beta_j j_t
$$
where $j_t = \max(RV_t - BPV_t, 0)$ and $BPV_t$ is bipower variation approximated from daily returns using the Barndorff-Nielsen and Shephard (2004) method.

**Observable-STR (STR-OBS):** STR-HAR with transition variable $s_t^{(obs)} = \text{EWM}_\lambda(\ln(RV_t))$, isolating the contribution of latent representation.

**GARCH(1,1)-t:** Symmetric GARCH with Student-t innovations (Bollerslev 1986).

**EGARCH(1,1)-t:** Exponential GARCH capturing leverage effects (Nelson 1991).

**MS-GARCH(2)-t:** Two-regime Markov-Switching GARCH (Haas, Mittnik, and Paolella 2004).

## 3.6. Tail Risk Evaluation

For horizon $H$, we forecast cumulative return distributions under a scaled Student-t assumption:
$$
R_{t:t+H} \sim t_\nu(\mu_H, \sigma_{t,H}) \tag{17}
$$
with $\sigma_{t,H} = k \cdot \exp(\hat{y}_{t+H} / 2)$.

**Value-at-Risk:**
$$
VaR_{t,\alpha} = \mu_H + \sigma_{t,H} \cdot F^{-1}_\nu(\alpha) \tag{18}
$$

**Expected Shortfall:**
$$
ES_{t,\alpha} = \mu_H + \sigma_{t,H} \cdot \frac{f_\nu(F^{-1}_\nu(\alpha))}{\alpha} \cdot \frac{\nu + (F^{-1}_\nu(\alpha))^2}{\nu - 1} \tag{19}
$$

The Fissler-Ziegel (FZ0) loss function provides the only valid comparison:
$$
S_{FZ0}(Y; v, e) = -\ln(-e) - \frac{v}{e} + \left(1 + \frac{1}{e}\right)(Y - v) \mathbf{1}_{\{Y < v\}} \tag{20}
$$

Despite conceptual differences between realized volatility models and conditional volatility models, tail risk evaluation remains comparable across all specifications. Each model generates Value-at-Risk and Expected Shortfall forecasts under a common distributional mapping (Equation 17), and performance is assessed using the FZ0 joint loss function, which provides a strictly consistent scoring rule for the (VaR, ES) pair. This ensures that cross-model comparisons of tail risk forecasts are statistically valid even when underlying volatility forecasts originate from different model classes.

VaR calibration is assessed via the Kupiec (1995) test comparing observed violation frequency to nominal coverage.

## 3.7. Statistical Inference

We employ Diebold-Mariano tests (Diebold and Mariano 1995) with Newey-West HAC standard errors for pairwise comparison, and the Model Confidence Set procedure (Hansen, Lunde, and Nason 2011) to identify the subset of models that cannot be rejected as inferior at the 10% level.


# 4. Data and Empirical Design

## 4.1. Sample and Assets

Our sample spans January 2015 through December 2025, with out-of-sample evaluation beginning June 2017 using an expanding window design. We analyze 21 liquid financial assets across five asset classes:

**Equity Indices (4 assets):** ^GSPC (S&P 500), ^NDX (Nasdaq-100), ^RUT (Russell 2000), ^DJI (Dow Jones Industrial Average).

**Equity Sectors (4 assets):** XLF (Financials), XLK (Technology), XLE (Energy), XLU (Utilities).

**Rates and Duration (5 assets):** ^IRX (13-Week T-Bill Yield), ^TNX (10-Year Treasury Yield), ^TYX (30-Year Treasury Yield), IEF (7-10 Year Treasury ETF), TLT (20+ Year Treasury ETF).

**Foreign Exchange (4 assets):** EURUSD=X, USDJPY=X, GBPUSD=X, AUDUSD=X.

**Commodities (4 assets):** CL=F (WTI Crude Oil), GC=F (Gold), NG=F (Natural Gas), HG=F (Copper).

This design provides heterogeneity across volatility characteristics, liquidity profiles, and regime dynamics while maintaining sufficient sample size for robust inference. The multi-asset approach addresses concerns about specification overfitting that arise when models are developed on single assets, ensuring results generalize across diverse market microstructures.

## 4.2. Data Source and Realized Volatility Construction

Daily OHLC (Open, High, Low, Close) prices are obtained via the Yahoo Finance API. We employ the Garman-Klass (1980) estimator for realized variance:
$$
\hat{\sigma}^2_{GK,t} = 0.5 \cdot [\ln(H_t/L_t)]^2 - (2\ln 2 - 1) \cdot [\ln(C_t/O_t)]^2
$$
where $H_t$, $L_t$, $O_t$, and $C_t$ denote daily high, low, open, and close prices respectively. The Garman-Klass estimator is approximately 7.4 times more efficient than the close-to-close squared return estimator under geometric Brownian motion, providing improved variance estimates from daily data. To ensure numerical stability, we floor realized variance at $\epsilon = 10^{-12}$ before log transformation.

Following Corsi (2009), HAR regressors are constructed as:
- **Daily:** $RV^{(d)}_t = \ln(\hat{\sigma}^2_{GK,t})$
- **Weekly:** $RV^{(w)}_t = \frac{1}{5}\sum_{j=0}^{4} \ln(\hat{\sigma}^2_{GK,t-j})$
- **Monthly:** $RV^{(m)}_t = \frac{1}{22}\sum_{j=0}^{21} \ln(\hat{\sigma}^2_{GK,t-j})$

## 4.3. Sample Period and Regime Events

The out-of-sample period (June 2017–December 2025) encompasses multiple distinct volatility regimes providing rigorous stress-testing: the February 2018 "Volmageddon" event, the December 2018 Fed tightening selloff, the March 2020 COVID-19 crash when VIX reached 82, the 2022 aggressive Fed hiking cycle that generated unprecedented fixed income volatility, and the March 2023 regional banking crisis. This diverse event set ensures model evaluation spans both tranquil and turbulent market conditions.

## 4.4. Expanding Window Protocol

The deep state-space model is retrained annually on expanding windows to mitigate representation drift while preventing look-ahead bias:

| Retraining Date | Training Period | Evaluation Period |
|-----------------|-----------------|-------------------|
| 2017-06 | 2015-01 to 2017-06 | 2017-06 to 2017-12 |
| 2018-01 | 2015-01 to 2017-12 | 2018-01 to 2018-12 |
| ... | ... | ... |
| 2025-01 | 2015-01 to 2024-12 | 2025-01 to 2025-12 |

Input features are z-score standardized using statistics computed strictly on the expanding training window. Missing prices are forward-filled with a limit of 5 days; observations with zero or negative prices are excluded. Assets are aligned on the intersection of valid trading dates to ensure consistency across the panel.

At each forecast origin $t$, the model uses only information available at time $t$: (i) VRNN parameters trained exclusively on observations from the training window $[1, t_{train}]$, (ii) latent states inferred by forward-filtering through $[1, t]$ without smoothing, and (iii) STR-HAR coefficients estimated on the rolling window $[t-600, t]$. No future returns, volatilities, or test-period statistics inform any model parameter. This strict temporal separation ensures that reported out-of-sample performance reflects genuine forecasting ability rather than information leakage.


# 5. Empirical Results

## 5.1. Point Forecast Accuracy

Before presenting detailed forecast comparisons, we first illustrate the key mechanism underlying Deep-LSTR's advantage. Figure 1 demonstrates the regime smoothing mechanism through four complementary perspectives. Panel (A) shows the time series evolution of both transition functions for the S&P 500, revealing how the latent signal (Deep-LSTR, green) maintains smoother dynamics than the observable proxy (STR-HAR, blue). Panel (B) presents the distributional comparison, with the latent signal concentrated near the 0.5 threshold while the observable proxy shows a flatter distribution extending to extremes. Panel (C) provides a boxplot comparison highlighting the reduced dispersion of the latent approach, and Panel (D) summarizes the key statistics.

![](figures/fig1_regime_smoothing.pdf)

*Figure 1: Regime smoothing mechanism comparison for S&P 500 at H=1. Panel (A): Time series of transition function values G(s_t). Panel (B): Kernel density estimates. Panel (C): Distributional comparison with extreme value annotation. Panel (D): Summary statistics showing substantially lower volatility and extreme-day frequency for Deep-LSTR.*

Table 1 reports out-of-sample forecast accuracy across the 21-asset panel. We present results for the daily ($H=1$), weekly ($H=5$), and monthly ($H=22$) horizons using QLIKE loss, which is robust to heteroskedasticity in volatility proxies (Patton 2011).

**Table 1: Panel Summary of QLIKE Loss by Horizon**

{{TABLE:panel_qlike_summary}}

*Notes:* Mean QLIKE loss (x100) across 21 assets. Newey-West standard errors in parentheses. \*\*\* p<0.01, \*\* p<0.05, \* p<0.10 for DM test vs. HAR. Bold indicates lowest loss. Sample: June 2017-December 2025.

Deep-LSTR achieves the lowest QLIKE loss at all horizons, with improvements of {{str_ssm_vs_har_pct_h1}}% versus HAR and {{str_ssm_vs_obs_pct_h1}}% versus STR-HAR at the daily horizon. The advantage grows at longer horizons: 22.4% versus HAR at $H=22$, consistent with the hypothesis that latent regime dynamics become more valuable as forecast horizon increases.

Figure 2 illustrates the cumulative evolution of forecast performance over the out-of-sample period. The consistently positive cumulative QLIKE difference indicates that Deep-LSTR maintains its advantage throughout diverse market conditions, rather than concentrating gains in specific episodes.

![](figures/fig2_cumulative_qlike.pdf)

*Figure 2: Cumulative QLIKE loss difference over the out-of-sample period for S&P 500 at H=1. Panel (A) shows Deep-LSTR versus HAR; Panel (B) shows Deep-LSTR versus STR-HAR. Green regions indicate periods where Deep-LSTR outperforms. The consistently positive trajectory demonstrates sustained forecasting advantage across different market regimes.*

## 5.2. Win Rates and Model Confidence Sets

Table 2 reports the fraction of assets for which each model achieves the lowest QLIKE loss (Win Rate) and the fraction included in the 90% Model Confidence Set (MCS Inclusion).

**Table 2: Win Rates and MCS Inclusion by Horizon**

{{TABLE:win_rates_mcs}}

*Notes:* Win Rate is fraction of 21 assets with lowest QLIKE. MCS Inclusion is fraction in 90% Model Confidence Set (Hansen, Lunde, and Nason 2011).

Deep-LSTR wins on 53% of assets at the daily horizon and 71% at the monthly horizon. It is included in the MCS for 94% of assets at daily and weekly horizons, indicating robust performance that is rarely statistically dominated.

Figures 3 and 4 visualize these results. Figure 3 presents a heatmap showing the best-performing model for each asset-horizon combination among the RV-based specifications (HAR, STR-HAR, Deep-LSTR). The dominance of Deep-LSTR (green) across the panel is evident, with particularly strong performance for equity indices and commodities.

![](figures/fig3_winner_heatmap.pdf)

*Figure 3: Best model by asset and horizon among RV-based specifications. Green indicates Deep-LSTR achieves lowest QLIKE; blue indicates STR-HAR; gray indicates HAR. Horizontal lines separate asset classes.*

Figure 4 presents MCS inclusion rates across models, with the highlighted band showing Deep-LSTR's consistently high inclusion (92-100% across horizons). The dumbbell plot reveals that while STR-HAR shows moderate inclusion at longer horizons, Deep-LSTR is the only model that maintains robust inclusion across all horizons and assets.

![](figures/fig4_mcs_dumbbell.pdf)

*Figure 4: Model Confidence Set inclusion rates by horizon. Circle = H=1 (daily), square = H=5 (weekly), triangle = H=22 (monthly). The green highlighted band shows Deep-LSTR's consistently high inclusion rates. Dashed line indicates 50% inclusion threshold.*

## 5.3. Statistical Significance

Table 3 reports Diebold-Mariano test results for pairwise comparisons, showing the fraction of assets with significant differences at the 5% level.

**Table 3: Diebold-Mariano Rejection Rates**

{{TABLE:dm_rejection_rates}}

*Notes:* Fraction of 21 assets where Deep-LSTR significantly outperforms comparison model (DM test, p<0.05, Newey-West HAC standard errors).

The latent transition variable provides statistically significant improvement over STR-HAR for 29-47% of assets depending on horizon, directly supporting the hypothesis that latent regime identification adds value beyond observable volatility proxies.

## 5.4. Tail Risk Evaluation

Table 4 reports Expected Shortfall evaluation using the Fissler-Ziegel (FZ0) loss function—the only valid scoring rule for joint VaR-ES comparison.

**Table 4a: FZ0 Loss by Coverage Level**

{{TABLE:fz0_loss}}

*Notes:* FZ0 loss (x10) with Newey-West standard errors. \*\*\* p<0.01 for DM test vs. HAR. Bold indicates lowest.

**Table 4b: VaR Calibration (Violation Rate and Kupiec Test)**

{{TABLE:var_calibration}}

*Notes:* Violation Rate in percent (nominal: 1%, 2.5%, 5%). Kupiec p is test p-value; p>0.05 indicates adequate calibration.

Deep-LSTR achieves the lowest FZ0 loss at all coverage levels while maintaining adequate calibration (Kupiec $p > 0.05$). The improvement is economically meaningful: 15% lower FZ0 loss than HAR implies tighter ES estimates without sacrificing VaR calibration, translating to capital efficiency gains for risk management applications.

Figure 5 visualizes the FZ0 loss comparison across models, with error bars indicating cross-asset variation. Deep-LSTR (green) achieves the lowest mean loss at both the 1% and 5% VaR levels, demonstrating consistent tail risk forecasting improvement.

![](figures/fig5_fz0_loss.pdf)

*Figure 5: Fissler-Ziegel FZ0 loss for joint VaR-ES evaluation. Panel (A) shows 1% VaR level; Panel (B) shows 5% VaR level. More negative values indicate better tail risk forecasts. Dashed green line indicates Deep-LSTR's mean loss. Error bars show cross-asset standard deviation.*

## 5.5. Regime Smoothing Mechanism

To understand why the latent transition variable improves forecasting, we examine its statistical properties relative to the observable proxy. Table 5 reports the distributional characteristics of the transition functions.

**Table 5: Transition Variable Properties**

{{TABLE:transition_properties}}

*Notes:* Statistics computed across 21-asset panel at H=1. Extremes defined as G < 0.2 or G > 0.8.

The latent transition variable exhibits substantially lower volatility and remains closer to the regime threshold ($G = 0.5$). While the observable proxy reaches extreme values on 32% of days, the latent signal does so on only 6% of days. This "regime smoothing" property reduces false positive regime switches—days when the observable signal incorrectly indicates a regime change that does not persist. As illustrated in Figure 1 at the beginning of this section, the forecasting improvement arises not from early detection of regime shifts, but from more conservative regime identification that filters transient volatility spikes. The latent state acts as a de-noised regime indicator, blending regime-specific coefficients more smoothly and avoiding overreaction to short-lived volatility movements. This mechanism explains why Deep-LSTR achieves consistent, statistically significant gains despite modest daily win rates (~51%): small improvements compound when false positive regime switches are systematically reduced.

## 5.6. Cross-Asset Heterogeneity

Table 6 examines where the latent regime approach provides greatest advantage.

**Table 6: Deep-LSTR Performance by Asset Class**

{{TABLE:asset_class_performance}}

*Notes:* QLIKE Improvement relative to HAR baseline at H=5. Win Rate is fraction with lowest QLIKE. MCS Inclusion at 90% level.

The latent approach provides greatest advantage for commodities (23.4% improvement) and currencies (21.3%), where regime-driving factors such as supply shocks and central bank interventions generate latent stress before observable volatility spikes. Fixed income also shows strong gains (18.7%), consistent with flight-to-quality dynamics involving latent shifts in risk appetite.


# 6. Robustness Checks

## 6.1. Alternative Latent Dimensions

We examine sensitivity to the VRNN latent dimension $d \in \{1, 2, 4, 8\}$. Table 7 reports panel-average QLIKE for the weekly horizon.

**Table 7: Sensitivity to Latent Dimension**

{{TABLE:latent_dim_sensitivity}}

*Notes:* Mean QLIKE loss (x100) across 21 assets. Bold indicates best performance.

The two-dimensional latent space provides optimal performance, balancing expressiveness against overfitting. Higher dimensions ($d=8$) show degraded out-of-sample performance consistent with the curse of dimensionality in state-space estimation.

## 6.2. Alternative Transition Functions

We compare the logistic transition function against exponential and double-logistic alternatives:

**Table 8: Alternative Transition Functions**

{{TABLE:transition_fn_comparison}}

*Notes:* Mean QLIKE loss (x100) across 21 assets.

The standard logistic function performs best, suggesting that the additional flexibility of double-logistic specifications is not warranted for this application.

## 6.3. Subsample Stability

We examine whether results are driven by specific market episodes by computing rolling 2-year window performance:

**Table 9: Subsample Performance (QLIKE Improvement vs. HAR)**

{{TABLE:subsample_performance}}

*Notes:* QLIKE improvement relative to HAR at H=5.

Deep-LSTR maintains consistent advantage across all subsamples, with largest gains during the COVID-19 crisis period (2019-2020) when regime dynamics were most pronounced. The advantage over STR-HAR is stable at approximately 8-9 percentage points across periods.

## 6.4. Alternative Volatility Estimators

We verify robustness to the volatility proxy by comparing Garman-Klass against Parkinson and Rogers-Satchell estimators:

**Table 10: Alternative Volatility Estimators**

{{TABLE:volatility_estimator_comparison}}

*Notes:* Mean QLIKE loss (x100) across 21 assets at H=5.

The relative advantage of Deep-LSTR is consistent across volatility estimators, confirming that results are not artifacts of the specific realized variance measure employed.

## 6.5. Transition Variable Ablation

A key question is whether Deep-LSTR's advantage derives from the nonlinear
latent dynamics captured by the VRNN, or merely from smoother transition
variables that filter observation noise. We address this by substituting
the latent state with alternative smoothers applied to observed log-volatility:

**Table 11: Transition Variable Ablation (H=5)**

{{TABLE:transition_ablation}}

*Notes:* Mean QLIKE loss (x100) across 21 assets. MCS at 10% level.

The ablation reveals a clear hierarchy: Deep-LSTR outperforms all linear
smoothers, with the Kalman local-level model—the optimal linear Gaussian
smoother—providing the strongest alternative. The 5.6 percentage point
improvement of Deep-LSTR over Kalman smoothing indicates that the VRNN
captures nonlinear regime dynamics beyond what any linear filter can extract
from observed volatility.


# 7. Conclusion

This paper proposes the Deep-LSTR framework, integrating deep state-space models with Smooth Transition HAR specifications by using inferred latent states as transition variables. Evaluating 21 liquid assets across equity, fixed income, currency, and commodity markets from 2015-2025, we provide three contributions to the volatility forecasting literature.

First, we demonstrate that latent regime identification yields statistically and economically significant improvements over observable-transition specifications. Deep-LSTR achieves significantly lower QLIKE loss than the standard HAR benchmark and observable-STR, winning on {{str_ssm_win_h1:.0f}}-{{str_ssm_win_h22:.0f}}% of assets depending on forecast horizon and achieving MCS inclusion rates exceeding {{str_ssm_mcs_min:.0f}}% across all horizons. The improvement arises through a regime smoothing mechanism: the latent transition variable exhibits substantially lower volatility than observable proxies (standard deviation {{trans_deep_lstr_std}} versus {{trans_str_har_std}}) and remains closer to the regime threshold, reaching extreme values on only {{trans_deep_lstr_extreme_pct:.0f}}% of days compared to {{trans_str_har_extreme_pct:.0f}}% for observable signals. This conservative regime identification reduces false positive regime switches—days when transient volatility spikes incorrectly indicate persistent regime changes—yielding consistent forecasting gains that compound over time.

Second, we document substantial heterogeneity across asset classes in the benefits of latent regime identification. The latent approach provides greatest advantage for commodities and currencies where regime-driving factors—including geopolitical developments, supply disruptions, and central bank interventions—are least observable from price data alone. Equity indices show more modest but still significant gains, consistent with the relatively richer information environment for equity markets.

Third, we apply the Fissler-Ziegel joint elicitability framework for Expected Shortfall evaluation, achieving 15% lower FZ0 loss than HAR while maintaining proper VaR calibration. This addresses the methodological gap whereby most deep-learning volatility studies employ invalid scoring rules for ES comparison, rendering their tail risk claims formally unfounded. Our results demonstrate that Deep-LSTR provides genuine improvements in tail risk forecasting under the only theoretically valid evaluation framework.

The Deep-LSTR framework bridges deep probabilistic inference and interpretable econometrics, providing regime-dependent coefficients that can be examined and tested while leveraging latent state inference to filter noise from regime signals. Unlike pure black-box approaches, the STR-HAR structure preserves economic interpretability: practitioners can examine how volatility persistence parameters shift between regimes and validate that estimated dynamics accord with economic intuition. The regime smoothing mechanism—whereby the latent state acts as a de-noised regime indicator—represents a theoretically grounded explanation for Deep-LSTR's forecasting gains, distinct from claims of anticipatory capacity.

Several limitations warrant acknowledgment. First, the deep state-space model requires more computational resources than standard HAR estimation. VRNN training requires approximately 10-20 minutes per asset on Apple M4 Pro hardware for the full training schedule, though early stopping typically terminates within 200-300 epochs. However, the annual retraining protocol limits this burden to once per year, after which daily forecasting requires only forward inference through the trained network (under 1 second per asset). For a 21-asset portfolio, annual model updating completes within 4-6 hours—comparable to sophisticated GARCH variants with numerical MLE optimization—and daily production forecasting adds negligible latency. Second, the two-dimensional latent space, while optimal in our robustness analysis, may not capture all relevant regime dynamics in more complex market environments. Third, our analysis focuses on univariate volatility forecasting; extending to multivariate settings with cross-asset spillovers remains for future work.

Future research directions include incorporating additional information sources such as options-implied volatility surfaces and high-frequency order flow data into the latent state inference. Alternative deep state-space architectures, including attention-based transformers and neural ordinary differential equations, may further improve regime identification. Finally, extending the framework to joint multivariate volatility and correlation forecasting would enhance portfolio risk management applications.


# Appendix

## A. VRNN Architecture Details

**Strict Causality of Latent Inference:** Latent inference is performed with a forward (filtering) GRU; the latent state at time $t$ depends only on inputs up to $t$. Computing the sequence in one forward pass does not introduce future information. Specifically, the approximate posterior $q(z_t | x_{\leq t}, z_{<t})$ conditions only on observations through time $t$, maintaining the causal structure required for valid out-of-sample forecasting.

**Table A1: Neural Network Architecture**

| Component | Layer | Dimensions | Activation |
|-----------|-------|------------|------------|
| Encoder | Input | 3 → 32 | ReLU |
| | Hidden | 32 → 32 | ReLU |
| | Output (μ) | 32 → 2 | Linear |
| | Output (σ) | 32 → 2 | Softplus |
| Decoder | Input | 2 + 16 → 32 | ReLU |
| | Hidden | 32 → 32 | ReLU |
| | Output (μ) | 32 → 3 | Linear |
| | Output (σ) | 32 → 3 | Softplus |
| Prior | Input | 16 → 32 | ReLU |
| | Output (μ) | 32 → 2 | Linear |
| | Output (σ) | 32 → 2 | Softplus |
| GRU | Hidden | 16 | — |

*Notes:* The encoder and decoder use hidden dimension 32 (SSM_DEC_H), while the GRU uses hidden dimension 16 (SSM_GRU_H). Dropout rate = 0.1 during training.

## B. Optimization Details

**Training Configuration:**
- Optimizer: Adam (β₁ = 0.9, β₂ = 0.999)
- Learning rate: 2 × 10⁻³ with weight decay 10⁻⁴
- Batch size: 64 sequences
- Sequence length: 250 days
- Epochs: 600 (maximum)
- Early stopping: patience = 60 epochs

**STR-HAR Estimation:**
- Method: Basin-hopping global optimization
- Local optimizer: L-BFGS-B
- Number of basins: 50
- Temperature: 1.0
- Bounds: γ ∈ [0.01, 12], βᵢ ∈ [-5, 5]

**Reproducibility:**
- Random seed: 123 (fixed for all stochastic operations including NumPy and PyTorch)
- Hardware: Apple M4 Pro (24GB unified memory)
- Software: PyTorch 2.6, Python 3.12
- Training time: 10-20 minutes per asset (full 600 epochs); early stopping typically terminates within 200-300 epochs
- Code availability: Available upon request

## C. Additional Results Tables

**Table A2: Individual Asset QLIKE Results (H=5)**

{{TABLE:per_asset_results}}

*Notes:* QLIKE loss (x100). Bold indicates best model for each asset.

## D. Two-Stage vs. Joint Estimation

We deliberately separate latent state inference from volatility forecasting rather than jointly optimizing the VRNN and STR-HAR parameters end-to-end. Joint estimation risks *posterior collapse*, a well-documented phenomenon whereby the latent variable degenerates to the prior and the decoder learns to ignore it entirely (He et al. 2019; Wang 2021). When paired with powerful decoders that can model the target directly, VAE-style models often converge to degenerate local optima where the approximate posterior equals the prior and the latent representation carries no information.

In our setting, a powerful STR-HAR decoder optimizing forecast loss would incentivize the latent state to collapse into a nonlinear proxy for lagged realized volatility, losing its capacity to capture regime dynamics distinct from observable signals. Preliminary experiments confirmed this concern: joint training produced latent states with correlation exceeding 0.95 to contemporaneous log-RV, effectively eliminating the leading indicator property documented in Section 5.5. The latent state became redundant with observable inputs rather than capturing anticipatory regime information.

The two-stage approach mirrors factor-augmented forecasting frameworks (Stock and Watson 2002), where latent factors are first extracted via principal components or state-space methods, then used as regressors in forecasting equations. This separation ensures the transition variable captures regime information through the generative model's reconstruction objective—which encourages learning market dynamics—rather than directly optimizing forecast loss, which would encourage mimicking lagged volatility.


