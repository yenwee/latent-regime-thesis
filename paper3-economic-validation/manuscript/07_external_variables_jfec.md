# 5. External Variables and Validation Design

This section describes the external variables used for validation and the design principles governing their selection. The variables are organized into categories that capture different dimensions of market stress. Crucially, all variables are held out from regime construction.

## 5.1 Categories of External Variables

We organize validation variables into four categories, each representing a distinct aspect of market conditions.

### 5.1.1 Funding Stress

Funding stress variables capture conditions in short-term credit markets. When funding is scarce or expensive, market participants face constraints that can amplify volatility and propagate stress across markets [@brunnermeier2009market].

Variables in this category include spreads between unsecured and secured lending rates, measures of interbank market tension, and indicators of dollar funding conditions. These variables are available at daily or higher frequency and exhibit substantial variation during stress episodes.

If latent regimes capture economically meaningful variation, we would expect alignment with funding stress measures during periods when funding conditions deteriorate.

### 5.1.2 Volatility Risk Premium

The volatility risk premium represents the compensation investors demand for bearing volatility risk [@bollerslev2009expected; @carr2009variance]. It is typically measured as the difference between implied volatility (from option prices) and realized volatility [@bekaert2014vix]. A high volatility risk premium suggests elevated risk aversion or uncertainty about future volatility.

Variables in this category include the VIX-RV spread for equity markets [@whaley2009understanding] and analogous measures for other asset classes where options are liquid. The volatility risk premium is forward-looking, reflecting expectations and risk preferences, while realized volatility is backward-looking.

Alignment between regimes and the volatility risk premium would suggest that regimes capture variation in risk compensation, not merely volatility levels.

### 5.1.3 Correlation and Dependence

Correlation measures capture the degree of co-movement across assets. During stress periods, correlations typically increase as diversification benefits decline [@longin2001extreme; @ang2002asymmetric]. This "correlation breakdown" is a hallmark of systemic stress [@forbes2002no].

Variables in this category include implied correlation indices, realized correlation measures computed from return data, and tail dependence estimates. These variables are particularly relevant for assessing whether the latent regime captures broad market stress dynamics that manifest across asset classes simultaneously.

### 5.1.4 Event-Based Stress Indicators

Event-based indicators capture discrete stress events rather than continuous measures. These include dummy variables for crisis periods, indicators of extreme market movements, and binary stress classifications from official sources.

These variables are useful for examining regime behavior during known stress episodes. They complement continuous measures by providing clear reference points for assessing regime dynamics.

## 5.2 Why Variables Are Held Out

The principle of holding out validation variables is essential to the analysis. If a variable were used in regime construction, any observed alignment would be uninformative---the regime would align with the variable by construction.

Our regime construction uses only realized volatility and, in some specifications, returns. It does not use:

- Option-implied measures (VIX, VVIX, skew indices)
- Funding market data (LIBOR, OIS, TED spread)
- Credit spreads (investment grade, high yield, CDS)
- Correlation measures (implied or realized)
- Macroeconomic indicators

This separation ensures that alignment, if observed, reflects underlying relationships rather than mechanical construction.

## 5.3 Criteria for Variable Inclusion

Variables are included in the validation set based on three criteria:

**Theoretical relevance.** The variable should be plausibly related to market stress conditions that might manifest in volatility regimes. We do not include variables with no theoretical connection to volatility dynamics.

**Data quality.** The variable should be available at sufficient frequency (daily or higher) with reliable measurement. Variables with substantial missing data, stale pricing, or known measurement issues are excluded.

**Independence from construction data.** The variable must not be mechanically related to realized volatility. This excludes transformations of realized volatility itself but permits variables that may be correlated with volatility through economic channels.

## 5.4 Validation Design

The validation design examines three types of relationships:

**Contemporaneous alignment.** We assess whether the regime state is correlated with external variables at the same point in time. This provides a baseline measure of association.

**Lead-lag relationships.** We examine whether regime states lead or lag external variables. Lead relationships are particularly informative: if regimes systematically precede stress indicator movements, this suggests the regime captures information not yet reflected in observables.

**Conditional distributions.** We compare the distribution of external variables in high-regime versus low-regime periods. Systematic differences indicate that regime states correspond to distinct market conditions.

The analysis does not impose a specific model relating regimes to external variables. We use descriptive statistics, correlations, and non-parametric comparisons rather than structural models. This approach is appropriate given our objective of assessing coherence rather than estimating causal effects.

## 5.5 Notation

Let $q_t \in [0,1]$ denote the regime indicator at time $t$, with higher values indicating the high-volatility regime. Let $x_{j,t}$ denote external variable $j$ at time $t$. We examine:

$$\rho(q_t, x_{j,t+k})$$

for various leads and lags $k$. We also compare:

$$\mathbb{E}[x_{j,t} | q_t > 0.5] \quad \text{vs.} \quad \mathbb{E}[x_{j,t} | q_t \leq 0.5]$$

and test whether these conditional means differ significantly.

Details of specific variables and sample periods are provided in the empirical results section.
