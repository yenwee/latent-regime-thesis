# Appendix

## A. Proof of ELBO Derivation

The evidence lower bound for the multivariate VRNN is derived as follows.

Starting from the log-likelihood of the observations:
$$
\log p(\mathbf{x}_{1:T}) = \log \int p(\mathbf{x}_{1:T}, \mathbf{z}_{1:T}) d\mathbf{z}_{1:T}
$$

Introducing the variational posterior $q(\mathbf{z}_{1:T} | \mathbf{x}_{1:T})$:
$$
\log p(\mathbf{x}_{1:T}) = \log \int \frac{q(\mathbf{z}_{1:T} | \mathbf{x}_{1:T})}{q(\mathbf{z}_{1:T} | \mathbf{x}_{1:T})} p(\mathbf{x}_{1:T}, \mathbf{z}_{1:T}) d\mathbf{z}_{1:T}
$$

Applying Jensen's inequality:
$$
\log p(\mathbf{x}_{1:T}) \geq \mathbb{E}_{q}\left[\log \frac{p(\mathbf{x}_{1:T}, \mathbf{z}_{1:T})}{q(\mathbf{z}_{1:T} | \mathbf{x}_{1:T})}\right] = \mathcal{L}(\mathbf{x}_{1:T})
$$

Under the VRNN factorization:
$$
\mathcal{L} = \sum_{t=1}^{T} \mathbb{E}_{q(\mathbf{z}_t)}[\log p(\mathbf{x}_t | \mathbf{z}_{\leq t}, \mathbf{x}_{<t})] - \text{KL}(q(\mathbf{z}_t | \mathbf{x}_{\leq t}) \| p(\mathbf{z}_t | \mathbf{z}_{<t}, \mathbf{x}_{<t}))
$$

## B. Multivariate VRNN Architecture Details

**Encoder Network:**
- Input: Concatenation of $\mathbf{x}_t \in \mathbb{R}^N$ and $\mathbf{h}_{t-1} \in \mathbb{R}^{32}$
- Hidden layer: 64 units, ReLU activation
- Output: $\boldsymbol{\mu}_{\text{enc}} \in \mathbb{R}^d$, $\log \boldsymbol{\sigma}^2_{\text{enc}} \in \mathbb{R}^d$

**Prior Network:**
- Input: $\mathbf{h}_{t-1} \in \mathbb{R}^{32}$
- Hidden layer: 64 units, ReLU activation
- Output: $\boldsymbol{\mu}_{\text{prior}} \in \mathbb{R}^d$, $\log \boldsymbol{\sigma}^2_{\text{prior}} \in \mathbb{R}^d$

**Decoder Network:**
- Input: $\mathbf{z}_t \in \mathbb{R}^d$
- Output: $\boldsymbol{\mu}_{\text{dec}} \in \mathbb{R}^N$, $\log \boldsymbol{\sigma}^2_{\text{dec}} \in \mathbb{R}^N$

**Recurrent Cell:**
- GRU with hidden dimension 32
- Input: Concatenation of $\mathbf{x}_t$ and $\mathbf{z}_t$

## C. Asset-Level Results

**Table A1: Full Asset-Level QLIKE Results**

{{TABLE:full_asset_qlike}}

*Notes: QLIKE loss for each asset at H=1, H=5, H=22. Lowest value in each row in bold.*

**Table A2: Asset-Level Diebold-Mariano Tests**

{{TABLE:asset_dm_tests}}

*Notes: p-values for Shared-VRNN vs. Asset-VRNN. Negative t-statistics indicate Shared-VRNN outperforms.*

## D. Additional Figures

**Figure A1: Regime State Time Series by Asset Class**

[Placeholder for figure showing inferred regime states separated by asset class]

**Figure A2: Transition Function Estimates**

[Placeholder for figure showing estimated $G(s_t; \gamma_i, c_i)$ curves for select assets]

**Figure A3: Rolling Window QLIKE Differences**

[Placeholder for figure showing time-varying performance differences]

## E. Data Details

**Table A3: Asset Summary Statistics**

{{TABLE:asset_summary_stats}}

*Notes: Sample period 2015-2024. RV in annualized volatility units. ADF is Augmented Dickey-Fuller test statistic.*

**Table A4: Cross-Asset Correlations**

{{TABLE:rv_correlations}}

*Notes: Pairwise correlations of daily log realized volatility.*

## F. Alternative Specifications

**Table A5: Alternative Loss Functions**

{{TABLE:alternative_losses}}

*Notes: MSE and MAE in addition to QLIKE. Results qualitatively similar across loss functions.*

**Table A6: Alternative Transition Functions**

{{TABLE:alternative_transitions}}

*Notes: Exponential and threshold transitions compared to logistic (baseline).*
