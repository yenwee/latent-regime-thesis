# Appendix

## A. VRNN Architecture Details

**Strict Causality of Latent Inference:** Latent inference is performed with a forward (filtering) GRU; the latent state at time $t$ depends only on inputs up to $t$. Computing the sequence in one forward pass does not introduce future information. Specifically, the approximate posterior $q(z_t | x_{\leq t}, z_{<t})$ conditions only on observations through time $t$, maintaining the causal structure required for valid out-of-sample forecasting.

**Table A1: Neural Network Architecture**

| Component | Layer | Dimensions | Activation |
|-----------|-------|------------|------------|
| GRU Encoder | Recurrent | 3 → 16 (hidden) | GRU gates |
| | Output ($\mu$) | 16 → 2 | Linear |
| | Output ($\log \sigma$) | 16 → 2 | Linear (clamped to $[-8, 4]$) |
| MLP Decoder | Input | 2 → 32 | Tanh |
| | Hidden | 32 → 32 | Tanh |
| | Output ($\hat{x}$) | 32 → 3 | Linear |
| Prior | Diagonal AR(1) | $z_t = \rho \odot z_{t-1} + \eta_t$ | $\rho = \tanh(\rho_{\text{raw}})$ |
| | Innovation $\sigma_\eta$ | 2 (learnable) | $\exp(\cdot)$ |
| Observation noise $\sigma_x$ | | 3 (learnable) | $\exp(\cdot)$ |

*Notes:* The GRU encoder directly produces approximate posterior parameters $(\mu, \sigma)$ for the latent state $z_t \in \mathbb{R}^2$. The prior is a diagonal AR(1) process with learnable persistence $\rho \in (-1, 1)$ and innovation variance, initialized at $\rho_{\text{raw}} = 0.3$. The decoder reconstructs observations from the latent state without conditioning on the GRU hidden state. No dropout is applied.

## B. Optimization Details

**Training Configuration:**
- Optimizer: Adam ($\beta_1 = 0.9$, $\beta_2 = 0.999$)
- Learning rate: $2 \times 10^{-3}$ with weight decay $10^{-4}$
- Gradient clipping: max norm = 5.0
- Training mode: full-sequence (single pass over entire training window per epoch)
- Epochs: 600 (maximum)
- Early stopping: patience = 60 epochs

**STR-HAR Estimation:**
- Method: Basin-hopping global optimization
- Local optimizer: L-BFGS-B
- Basin-hopping iterations: 5
- Temperature: 1.0
- Bounds: γ ∈ [0.01, 12], βᵢ ∈ [-5, 5]

**Parameter Bound Justification:** The smoothness parameter γ is constrained to [0.01, 12] following recommendations in the STR literature [@terasvirta1994specification; @vandijk2002smooth]. The lower bound γ = 0.01 prevents the transition function from degenerating to a constant (eliminating regime-switching behavior), while the upper bound γ = 12 prevents excessively abrupt transitions that would render the model numerically unstable and approach a threshold specification where gradient-based optimization fails. The coefficient bounds βᵢ ∈ [-5, 5] prevent parameter explosion during optimization while accommodating the typical range of HAR coefficients observed in realized volatility applications; preliminary estimation on our sample confirmed all converged coefficients fall well within these bounds.

**Hyperparameter Selection:**

The VRNN architecture employs a deliberately parsimonious design to prevent overfitting given the relatively short training windows (~600 observations per asset). The GRU hidden dimension (16 units) and decoder hidden dimension (32 units) were selected via preliminary grid search over {8, 16, 32, 64} on held-out validation loss, with smaller architectures preferred when validation performance was comparable. The learning rate (2 x 10^-3) and weight decay (10^-4) follow standard practices for Adam optimization in variational models. The KL divergence term in the ELBO and the AR(1) prior jointly regularize the latent dynamics, encouraging smooth temporal evolution without requiring explicit dropout. The two-dimensional latent space (d=2) was validated in Section 6.1, with higher dimensions showing degraded out-of-sample performance consistent with overfitting. This parsimonious architecture ensures the model generalizes across the diverse assets in our panel while remaining computationally tractable for annual retraining.

**Reproducibility:**
- Random seed: 123 (fixed for all stochastic operations including NumPy and PyTorch)
- Hardware: Apple M4 Pro (24GB unified memory)
- Software: PyTorch 2.6, Python 3.12
- Training time: 10-20 minutes per asset (full 600 epochs); early stopping typically terminates within 200-300 epochs

**Code and Data Availability:**
Replication code is publicly available at https://github.com/yenwee/latent-regime-thesis. The repository includes deterministic scripts to download all public data used in the analysis, a locked software environment (`requirements-lock.txt`) with pinned package versions for exact reproducibility, and configuration files reproducing all reported results.

**CUDA Determinism Settings:**
All experiments were conducted on CPU, which provides deterministic behavior given fixed random seeds. For users seeking to replicate results on CUDA-enabled GPUs, the following determinism settings should be enabled:

```python
import torch
import os

# Set CUDA determinism environment variable
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Enable deterministic algorithms globally
torch.use_deterministic_algorithms(True)

# Ensure cuDNN determinism
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

These settings ensure reproducible results across runs at the cost of potential performance degradation. The `CUBLAS_WORKSPACE_CONFIG` environment variable must be set before any CUDA operations. Note that some PyTorch operations lack deterministic implementations on CUDA; if encountered, users may need to replace these operations with deterministic alternatives or accept minor numerical variations. For exact replication of reported results, we recommend using CPU execution as in the original experiments.

## C. Additional Results Tables

**Table A2: Individual Asset QLIKE Results (H=5)**

| Asset | Class | HAR | STR-OBS | Deep-LSTR | GARCH-t | EGARCH-t | Best |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| Dow Jones | Equity | -9.091 | -9.091 | **-9.093** | -8.855 | -8.856 | Deep-LSTR |
| S&P 500 | Equity | -9.111 | -9.111 | **-9.115** | -8.829 | -8.829 | Deep-LSTR |
| NASDAQ-100 | Equity | -8.504 | -8.504 | **-8.506** | -8.230 | -8.238 | Deep-LSTR |
| Russell 2000 | Equity | -8.446 | -8.447 | **-8.448** | -8.150 | -8.191 | Deep-LSTR |
| Energy | Sector | -7.818 | -7.818 | **-7.820** | -7.650 | -7.650 | Deep-LSTR |
| Financials | Sector | -8.485 | -8.486 | **-8.488** | -8.270 | -8.281 | Deep-LSTR |
| Technology | Sector | -8.304 | -8.305 | **-8.306** | -8.060 | -8.072 | Deep-LSTR |
| Utilities | Sector | -8.429 | -8.430 | **-8.430** | -8.332 | -8.325 | Deep-LSTR |
| 7-10Y Bond | Fixed Inc. | -11.209 | -11.210 | **-11.211** | -10.680 | -10.741 | Deep-LSTR |
| 20+Y Bond | Fixed Inc. | -9.472 | -9.472 | **-9.473** | -9.026 | -9.083 | Deep-LSTR |
| 10Y Treasury | Fixed Inc. | -7.443 | **-7.539** | -7.524 | -7.186 | -7.248 | STR-OBS |
| 30Y Treasury | Fixed Inc. | -7.838 | **-7.931** | -7.910 | -7.641 | -7.696 | STR-OBS |
| AUD/USD | Currency | -9.199 | -9.206 | **-9.206** | -9.052 | -8.979 | Deep-LSTR |
| EUR/USD | Currency | -9.884 | -9.886 | **-9.887** | -9.787 | -9.784 | Deep-LSTR |
| GBP/USD | Currency | -9.540 | **-9.541** | -9.541 | -9.459 | -9.447 | STR-OBS |
| USD/JPY | Currency | -9.669 | -9.672 | **-9.674** | -9.544 | -9.562 | Deep-LSTR |
| Crude Oil | Commodity | -6.789 | -6.790 | **-6.791** | -6.682 | -6.675 | Deep-LSTR |
| Gold | Commodity | -8.957 | **-8.992** | -8.820 | -8.909 | -8.898 | STR-OBS |
| Copper | Commodity | -8.322 | -8.485 | **-8.496** | -8.093 | -8.083 | Deep-LSTR |
| Natural Gas | Commodity | -5.970 | -5.971 | **-5.972** | -5.845 | -5.857 | Deep-LSTR |

*Notes:* QLIKE loss (x100). Bold indicates best model for each asset.

## D. Two-Stage vs. Joint Estimation

We deliberately separate latent state inference from volatility forecasting rather than jointly optimizing the VRNN and STR-HAR parameters end-to-end. Joint estimation risks *posterior collapse*, a well-documented phenomenon whereby the latent variable degenerates to the prior and the decoder learns to ignore it entirely [@he2019lagging; @wang2021posterior]. When paired with powerful decoders that can model the target directly, VAE-style models often converge to degenerate local optima where the approximate posterior equals the prior and the latent representation carries no information.

In our setting, a powerful STR-HAR decoder optimizing forecast loss would incentivize the latent state to collapse into a nonlinear proxy for lagged realized volatility, losing its capacity to capture regime dynamics distinct from observable signals. Preliminary experiments confirmed this concern: joint training produced latent states with correlation exceeding 0.95 to contemporaneous log-RV, effectively eliminating the leading indicator property documented in Section 5.5. The latent state became redundant with observable inputs rather than capturing anticipatory regime information.

The two-stage approach mirrors factor-augmented forecasting frameworks [@stock2002forecasting], where latent factors are first extracted via principal components or state-space methods, then used as regressors in forecasting equations. This separation ensures the transition variable captures regime information through the generative model's reconstruction objective—which encourages learning market dynamics—rather than directly optimizing forecast loss, which would encourage mimicking lagged volatility.

## E. Data and Code Availability

Daily OHLC price data were obtained from the Yahoo Finance API (https://finance.yahoo.com/). The dataset covers 21 liquid financial assets across equity indices, equity sectors, fixed income, foreign exchange, and commodity markets from January 2015 through December 2025.

Replication code implementing the Deep-LSTR framework, including VRNN training, STR-HAR estimation, and all evaluation procedures, is publicly available at https://github.com/yenwee/latent-regime-thesis. The repository includes:
- Python implementation of the VRNN architecture
- STR-HAR estimation routines
- Evaluation scripts for QLIKE, FZ0 loss, and Model Confidence Set procedures
- Configuration files reproducing all reported results
- Deterministic scripts to download all public data
- Locked software environment for reproducibility

Raw price data can be obtained directly from Yahoo Finance using the provided asset tickers.
