# 4. Model

This section presents the model architecture. We first describe the multivariate VRNN that infers the shared latent regime state, then the asset-specific STR-HAR forecasting layer that uses this state, and finally the training procedure.

## 4.1 Multivariate VRNN for Shared Latent States

The core innovation is extending the Variational Recurrent Neural Network to multivariate observations. At each time $t$, the model observes realized volatility from $N$ assets: $\mathbf{x}_t = (RV_{1,t}, RV_{2,t}, \ldots, RV_{N,t})'$. The goal is to infer a shared latent state $z_t \in \mathbb{R}^d$ that captures the common regime structure.

The generative model factorizes as:
$$
p(\mathbf{x}_{1:T}, \mathbf{z}_{1:T}) = \prod_{t=1}^{T} p(\mathbf{x}_t | \mathbf{z}_{\leq t}, \mathbf{x}_{<t}) \, p(\mathbf{z}_t | \mathbf{z}_{<t}, \mathbf{x}_{<t})
$$

The latent state follows a Gaussian prior conditioned on past observations and latent states:
$$
p(\mathbf{z}_t | \mathbf{z}_{<t}, \mathbf{x}_{<t}) = \mathcal{N}\left(\boldsymbol{\mu}_{\text{prior},t}, \text{diag}(\boldsymbol{\sigma}^2_{\text{prior},t})\right)
$$
where the prior parameters are computed by a neural network from a recurrent hidden state $\mathbf{h}_{t-1}$ that summarizes past information.

The key structural difference from asset-specific VRNNs is that the latent state $\mathbf{z}_t$ is shared across all assets. The encoder observes the full vector $\mathbf{x}_t$ rather than individual asset volatilities:
$$
q(\mathbf{z}_t | \mathbf{z}_{<t}, \mathbf{x}_{\leq t}) = \mathcal{N}\left(\boldsymbol{\mu}_{\text{enc},t}, \text{diag}(\boldsymbol{\sigma}^2_{\text{enc},t})\right)
$$
where encoder parameters are computed from both $\mathbf{x}_t$ and $\mathbf{h}_{t-1}$.

The recurrent hidden state updates as:
$$
\mathbf{h}_t = f_{\text{RNN}}(\mathbf{h}_{t-1}, \mathbf{x}_t, \mathbf{z}_t)
$$
using a GRU cell that takes the stacked volatility observation and sampled latent state as input.

This architecture has the property that all assets contribute to the inference of $\mathbf{z}_t$. When asset $i$ shows a large volatility spike, this provides evidence about the shared regime state that affects the latent distribution for all assets. The cross-sectional averaging implicit in the encoder stabilizes regime identification relative to single-asset models.

## 4.2 STR-HAR Forecasting Layer

Given the shared latent state, each asset has its own forecasting equation. We use the STR-HAR specification where the latent state governs transitions between regimes:
$$
RV_{i,t+h} = (1 - G(s_t; \gamma_i, c_i)) \cdot f^{(L)}_i(\mathbf{x}^{(i)}_t) + G(s_t; \gamma_i, c_i) \cdot f^{(H)}_i(\mathbf{x}^{(i)}_t) + \varepsilon_{i,t+h}
$$

Here $s_t = g(\mathbf{z}_t)$ is a scalar summary of the latent state (e.g., the first principal component or a learned linear projection), $G(\cdot)$ is a logistic transition function, and $f^{(L)}_i$ and $f^{(H)}_i$ are low and high regime HAR components for asset $i$:
$$
f^{(r)}_i(\mathbf{x}^{(i)}_t) = \beta^{(r)}_{i,0} + \beta^{(r)}_{i,d} RV^{(d)}_{i,t} + \beta^{(r)}_{i,w} RV^{(w)}_{i,t} + \beta^{(r)}_{i,m} RV^{(m)}_{i,t}
$$

The transition function takes the standard logistic form:
$$
G(s_t; \gamma_i, c_i) = \frac{1}{1 + \exp(-\gamma_i(s_t - c_i))}
$$

Crucially, while the latent state $s_t$ is shared across assets, the transition parameters $(\gamma_i, c_i)$ and HAR coefficients $(\beta^{(L)}_i, \beta^{(H)}_i)$ are asset-specific. This implements partial pooling: the timing of regime transitions is informed by cross-asset evidence, but the response to those transitions varies by asset.

The parameterization allows for heterogeneous regime sensitivity. An asset with high $\gamma_i$ transitions sharply between regimes; one with low $\gamma_i$ transitions gradually. An asset with high $c_i$ requires a more extreme latent state to enter the high regime. These differences capture the empirical heterogeneity in how assets respond to systemic stress.

## 4.3 Decoder and Reconstruction

The decoder maps from the latent state to the observed volatility distribution. Given $\mathbf{z}_t$ and $\mathbf{h}_{t-1}$, the decoder produces asset-specific distribution parameters:
$$
p(RV_{i,t} | \mathbf{z}_t, \mathbf{h}_{t-1}) = \mathcal{N}(\mu^{\text{dec}}_{i,t}, (\sigma^{\text{dec}}_{i,t})^2)
$$

The decoder architecture allows for asset-specific mappings from the shared latent state to observation distributions. This is distinct from the forecasting layer: the decoder explains current observations given the latent state, while the STR-HAR layer forecasts future volatility.

## 4.4 Training Objective

The model is trained by maximizing the evidence lower bound (ELBO):
$$
\mathcal{L} = \sum_{t=1}^{T} \left[ \mathbb{E}_{q(\mathbf{z}_t)}[\log p(\mathbf{x}_t | \mathbf{z}_t, \mathbf{h}_{t-1})] - \beta \cdot \text{KL}(q(\mathbf{z}_t) \| p(\mathbf{z}_t)) \right] - \lambda \cdot \mathcal{L}_{\text{forecast}}
$$

The first term is the reconstruction likelihood summed across assets. The KL divergence term regularizes the approximate posterior toward the prior, with $\beta$ controlling the strength of regularization. The forecasting loss $\mathcal{L}_{\text{forecast}}$ is the out-of-sample prediction error from the STR-HAR layer, encouraging the latent state to be informative for forecasting.

We use a two-phase training procedure. In the first phase, the VRNN is trained on reconstruction alone to learn meaningful latent representations. In the second phase, the forecasting layer is added and the full model is trained end-to-end, allowing the latent state to adapt to the forecasting objective.

## 4.5 Benchmark Comparisons

We compare the shared latent regime model against several alternatives:

**Asset-specific VRNN**: Each asset has its own VRNN and STR-HAR layer, with no cross-asset information. This is the baseline from our companion paper.

**PCA regime**: The first principal component of the realized volatility panel serves as an observable transition variable. This tests whether cross-asset averaging alone, without latent modeling, captures regime structure.

**VIX proxy**: The VIX index serves as the transition variable for all assets. This tests whether an established observable stress indicator suffices.

**Cross-asset average**: The cross-sectional average of realized volatility serves as the transition variable. This is a simple observable alternative to the latent state.

These comparisons isolate the contribution of (a) cross-asset information, (b) latent versus observable regime indicators, and (c) deep versus simple averaging.

## 4.6 Implementation Details

The multivariate VRNN uses a latent dimension of $d=4$ and GRU hidden dimension of 32. The encoder and prior networks are two-layer MLPs with ReLU activation. The decoder is a linear mapping from latent state to observation parameters.

Training uses the Adam optimizer with learning rate $10^{-3}$ and batch size 32. We use gradient clipping to stabilize training. The model is implemented in PyTorch and trained on a single GPU.

For the STR-HAR layer, we initialize the transition parameters at $\gamma_i = 5$ and $c_i = 0$ (centered transition). The HAR coefficients are initialized from OLS estimates of a standard HAR model.
