# 3. Methodology

The Deep Latent State-Space Smooth Transition Autoregressive (Deep-LSTR) model integrates three components into an end-to-end differentiable system: (i) a Variational Recurrent Neural Network that infers a continuous latent state $z_t \in \mathbb{R}^d$ from observed market features; (ii) a linear projection layer mapping the latent state to a scalar transition variable $s_t = \alpha^\top z_t$; and (iii) a Smooth Transition HAR forecasting equation where regime-dependent coefficients are governed by the inferred $s_t$. This architecture preserves the interpretability of classical econometric models while leveraging deep probabilistic inference to construct a smoothed transition variable that filters transient volatility noise from true regime information.

## 3.1. Target Variable: Range-Based Variance Proxy

We employ the Garman-Klass (GK) estimator [@garman1980estimation] as our range-based variance proxy:
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

Financial markets violate linearity and Gaussianity through volatility clustering, leverage effects, fat tails, and regime-dependent dynamics. The VRNN [@chung2015recurrent] relaxes these restrictions by parameterizing transition and emission functions with neural networks while preserving the separation between latent dynamics and observations:
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
where $\mu_\theta, \sigma_\theta$ are neural network functions with $h_t$ the GRU hidden state. The prior is initialized as an AR(1) process, representing the simplest stationary specification that captures temporal dependence in latent dynamics [@harvey1989forecasting; @durbin2012time]. This choice provides principled regularization toward smooth temporal evolution while allowing the posterior to deviate when warranted by observed data, a standard approach in variational recurrent architectures [@chung2015recurrent].

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

We deliberately separate latent state inference from volatility forecasting. Joint end-to-end estimation risks degeneracy whereby the latent state collapses into a nonlinear proxy for realized volatility, losing its capacity to capture regime dynamics distinct from observable signals. Our two-stage approach mirrors factor-augmented forecasting frameworks [@stock2002forecasting], ensuring the transition variable captures regime information through the generative model rather than directly optimizing forecast loss. This separation also facilitates diagnostic analysis: we can examine the statistical properties of the inferred latent state relative to observable volatility proxies, particularly its smoothing characteristics that distinguish it from noisier observable signals.

We optimize using Adam with learning rate $2 \times 10^{-3}$ and weight decay $10^{-4}$ for up to 600 epochs with early stopping (patience of 60 epochs), employing the reparameterization trick [@kingma2014auto] for gradient-based optimization through stochastic sampling.

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

A potential concern with Equation (13) is circularity: if $s_t$ is trained to mimic noisy $\ln(RV_t)$, does this reintroduce the measurement noise we claim to filter? The resolution lies in the information bottleneck principle [@tishby2000information]. The ELBO objective (Equation 11) includes a KL-divergence term that regularizes the approximate posterior $q(z_t|x_{\leq t})$ toward the prior $p(z_t|z_{t-1})$. This regularization acts as a capacity constraint on the latent channel: the mutual information $I(X; Z)$ is upper-bounded by the KL divergence term [@alemi2017deep]. Consequently, $z_t$ cannot encode arbitrary high-frequency fluctuations present in the input---it must selectively compress information about the volatility process while discarding idiosyncratic noise. The linear projection $\alpha^\top z_t$ therefore extracts a smoothed signal from an already bandwidth-limited representation, rather than fitting to raw noise. This mechanism parallels the $\beta$-VAE framework [@higgins2017beta], where increasing the KL weight forces the latent representation to discard irrelevant variation. In our context, the AR(1) prior on $z_t$ combined with KL regularization creates a temporal smoothness prior that preferentially encodes persistent volatility regimes over transient fluctuations.

The projection is economically motivated: while market regimes are driven by high-dimensional processes, their impact on volatility dynamics is effectively one-dimensional, consistent with principal component analyses finding that a single factor explains 70-90% of cross-sectional variance in realized volatilities [@andersen2001distribution]. Standardization of $z_t$ and the OLS fit in Equation (13) use only training-period data; the projection is then applied out-of-sample without refitting. Thus, $s_t$ for periods $t > T_{train}$ is a genuine ex-ante signal derived from the compressed representation, not a fitted transformation of future realized volatility.

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

We distinguish between two classes of benchmarks. The primary comparison is conducted within the realized volatility literature: the standard HAR model [@corsi2009simple] and the observable-transition STR-HAR serve as direct competitors employing the same forecasting target. Additionally, we include conditional volatility models—GARCH, EGARCH, and MS-GARCH—as external benchmarks from the return-based volatility literature. These models forecast the conditional variance of returns rather than realized volatility measures constructed from high-frequency or range-based data. Their inclusion provides context by assessing whether latent regime identification offers improvements relative to widely used return-based alternatives, while acknowledging that differences in information sets and forecasting objectives naturally favor realized-measure-based models when the evaluation target is realized volatility.

**Linear HAR:** Standard @corsi2009simple specification estimated via OLS on a rolling 600-day window.

**LHAR (Leverage HAR):** Extends HAR with an asymmetric leverage term capturing the well-documented phenomenon that negative returns increase future volatility more than positive returns of equal magnitude [@corsi2012discrete]:
$$
\hat{y}_{t+H} = \beta_0 + \beta_d x^{(d)}_t + \beta_w x^{(w)}_t + \beta_m x^{(m)}_t + \beta_{lev} r^{-}_t
$$
where $r^{-}_t = \min(r_t, 0)$.

**HAR-J (Jump HAR):** Separates realized variance into continuous and jump components following @andersen2007roughing:
$$
\hat{y}_{t+H} = \beta_0 + \beta_d x^{(d)}_t + \beta_w x^{(w)}_t + \beta_m x^{(m)}_t + \beta_j j_t
$$
where $j_t = \max(RV_t - BPV_t, 0)$ and $BPV_t$ is bipower variation approximated from daily returns using the @barndorffnielsen2004power method.

**HAR-CJ (Continuous-Jump HAR):** Extends HAR-J by modeling continuous and jump components separately at daily, weekly, and monthly horizons following @andersen2007roughing:
$$
\hat{y}_{t+H} = \beta_0 + \beta_C^d C^{(d)}_t + \beta_C^w C^{(w)}_t + \beta_C^m C^{(m)}_t + \beta_J^d J^{(d)}_t + \beta_J^w J^{(w)}_t + \beta_J^m J^{(m)}_t
$$
where the continuous component $C_t = \min(RV_t, BPV_t)$ and jump component $J_t = \max(RV_t - BPV_t, 0)$, with weekly and monthly aggregations computed analogously to the standard HAR components. The min/max formulation ensures non-negative components and addresses finite-sample estimation error where $BPV_t$ may occasionally exceed $RV_t$.

LHAR, HAR-J, and HAR-CJ were evaluated in preliminary analysis but excluded from the final comparison: all three produced QLIKE losses within 0.02% of the standard HAR across the panel, providing negligible incremental information given our use of range-based (rather than tick-level) volatility estimators that do not separately identify jumps. The standard HAR therefore serves as a sufficient linear benchmark.

**Observable-STR (STR-OBS):** STR-HAR with transition variable $s_t^{(obs)} = \text{EWM}_\lambda(\ln(RV_t))$, isolating the contribution of latent representation.

**GARCH(1,1)-t:** Symmetric GARCH with Student-t innovations [@bollerslev1986generalized].

**EGARCH(1,1)-t:** Exponential GARCH capturing leverage effects [@nelson1991conditional].

**MS-GARCH(2)-t:** Two-regime Markov-Switching GARCH [@haas2004new].

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

The @fissler2016higher FZ0 loss function provides the only valid comparison:
$$
S_{FZ0}(Y; v, e) = -\ln(-e) - \frac{v}{e} + \left(1 + \frac{1}{e}\right)(Y - v) \mathbf{1}_{\{Y < v\}} \tag{20}
$$

Despite conceptual differences between realized volatility models and conditional volatility models, tail risk evaluation remains comparable across all specifications. Each model generates Value-at-Risk and Expected Shortfall forecasts under a common distributional mapping (Equation 17), and performance is assessed using the FZ0 joint loss function, which provides a strictly consistent scoring rule for the (VaR, ES) pair. This ensures that cross-model comparisons of tail risk forecasts are statistically valid even when underlying volatility forecasts originate from different model classes.

VaR calibration is assessed via the @kupiec1995techniques test comparing observed violation frequency to nominal coverage.

## 3.7. Statistical Inference

We employ Diebold-Mariano tests [@diebold1995comparing] with Newey-West HAC standard errors for pairwise comparison, and the Model Confidence Set procedure [@hansen2011model] to identify the subset of models that cannot be rejected as inferior at the 10% level.

**HAC Lag Selection.** The Newey-West estimator requires specification of a truncation lag $L$ for the Bartlett kernel. Following the recommendation of @diebold1995comparing for multi-step forecasts, we set $L = \max(20, 2H)$ where $H$ denotes the forecast horizon. This choice ensures adequate accommodation of the MA($H-1$) structure induced by overlapping forecast errors while providing a minimum lag of 20 to capture additional serial dependence from model misspecification and estimation error. The Bartlett kernel weights are $w_l = 1 - l/(L+1)$ for $l = 1, \ldots, L$.
