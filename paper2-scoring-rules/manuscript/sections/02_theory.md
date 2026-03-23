# 2. Theoretical Framework

This section develops the formal theory of regime-conditional proper scoring rules for volatility forecasts. We begin with a review of elicitability and proper scoring rules in Section 2.1, drawing on the foundational results of @gneiting2011making and @patton2011volatility. Section 2.2 introduces conditional elicitability under regime-switching and establishes that strict consistency is preserved under regime conditioning. Section 2.3 derives conditions under which unconditional scoring rules produce ranking reversals---cases where the unconditional winner is the conditional loser in the dominant regime. Section 2.4 establishes sufficient conditions for unconditional and conditional rankings to agree. Section 2.5 extends the Model Confidence Set of @hansen2011model to regime-conditional settings. Section 2.6 treats the joint elicitability of Value-at-Risk and Expected Shortfall under regime conditioning, building on @fissler2016higher. Section 2.7 connects the framework to the conditional predictive ability tests of @giacomini2006tests, addressing the additional complications that arise when the conditioning event involves a latent state.

**Notation.** Throughout, $n$ denotes the sample size and $T: \mathcal{F} \to \mathbb{R}$ denotes a statistical functional. We reserve $T$ exclusively for functionals to avoid ambiguity with the sample size.

## 2.1 Elicitability and Proper Scoring Rules

We begin by establishing notation and reviewing the core results from the elicitability literature. Let $Y$ denote a real-valued random variable with distribution $F \in \mathcal{F}$, where $\mathcal{F}$ is a convex class of probability distributions on $\mathbb{R}$. A statistical functional $T: \mathcal{F} \to \mathbb{R}$ maps distributions to real numbers; familiar examples include the mean $T(F) = \mathbb{E}_F[Y]$, the variance $T(F) = \text{Var}_F(Y)$, and quantiles $T(F) = F^{-1}(\alpha)$.

A scoring function $S: \mathbb{R} \times \mathbb{R} \to [0, \infty)$ maps a point forecast $\hat{y}$ and a realized value $y$ to a non-negative loss. The scoring function is said to be *consistent* for $T$ if the expected score is minimized at the true functional value, and *strictly consistent* if the minimizer is unique. These properties are formalized in the following definition, which follows @gneiting2011making.

**Definition 1** (Consistency and strict consistency). A scoring function $S: \mathbb{R} \times \mathbb{R} \to [0,\infty)$ is *consistent* for the functional $T: \mathcal{F} \to \mathbb{R}$ relative to $\mathcal{F}$ if, for all $F \in \mathcal{F}$ and all $x \in \mathbb{R}$,
$$
\mathbb{E}_F[S(T(F), Y)] \leq \mathbb{E}_F[S(x, Y)].
$$
The scoring function is *strictly consistent* for $T$ if equality holds only when $x = T(F)$.

The concept of elicitability, introduced formally by @lambert2008eliciting and developed extensively by @gneiting2011making, characterizes which functionals admit strictly consistent scoring functions.

**Definition 2** (Elicitability). A functional $T: \mathcal{F} \to \mathbb{R}$ is *elicitable* relative to $\mathcal{F}$ if there exists a scoring function $S$ that is strictly consistent for $T$ relative to $\mathcal{F}$.

@gneiting2011making showed that the mean, quantiles, and expectiles are elicitable, while the variance is elicitable as a function of the first two moments. Critically for volatility forecasting, @gneiting2011making established that the variance functional $T(F) = \text{Var}_F(Y) = \mathbb{E}_F[Y^2] - (\mathbb{E}_F[Y])^2$ is not directly elicitable as a standalone functional but becomes elicitable when the mean is also specified. @patton2011volatility resolved this for the volatility forecasting setting by showing that the conditional expectation of a volatility proxy is elicitable, and that certain loss functions maintain consistent model rankings even when the true volatility is unobservable.

The loss function that is central to our analysis is the QLIKE loss, which belongs to the Bregman divergence family. For a volatility forecast $\hat{\sigma}^2$ and a realized proxy $\sigma^2$, the QLIKE loss is defined as
$$
S_{\text{QLIKE}}(\hat{\sigma}^2, \sigma^2) = \frac{\sigma^2}{\hat{\sigma}^2} - \log\frac{\sigma^2}{\hat{\sigma}^2} - 1.
$$

@patton2011volatility proved that $S_{\text{QLIKE}}$ is strictly consistent for the conditional variance under mild regularity conditions. Moreover, and crucially for empirical work, @patton2011volatility showed that the ranking of forecasts by expected QLIKE loss is *robust to noise in the volatility proxy*: if we replace the latent true variance with a conditionally unbiased proxy (such as realized variance computed from high-frequency data), the ranking of forecasts is preserved. This robustness property makes QLIKE the workhorse loss function for volatility forecast comparison. We note that proxy robustness requires conditional unbiasedness of the proxy with respect to the information set on which the forecasts are based. When we condition on regime state, this requires that the proxy remain conditionally unbiased within each regime, a condition that is satisfied when the proxy is constructed from intraday data that are available regardless of the regime state.

More broadly, @patton2011volatility established that any member of the homogeneous Bregman divergence family,
$$
S_b(\hat{\sigma}^2, \sigma^2) = \begin{cases} \frac{1}{(b+1)(b+2)}\left[\sigma^{2(b+2)} - \hat{\sigma}^{2(b+2)}\right] - \frac{1}{b+1}\hat{\sigma}^{2(b+1)}(\sigma^2 - \hat{\sigma}^2) & b \neq -1, -2 \\ \frac{\sigma^2}{\hat{\sigma}^2} - \log\frac{\sigma^2}{\hat{\sigma}^2} - 1 & b = -2 \\ \sigma^2 \log\frac{\sigma^2}{\hat{\sigma}^2} + \hat{\sigma}^2 - \sigma^2 & b = -1 \end{cases}
$$
yields a scoring function that is robust to proxy noise, with QLIKE corresponding to $b = -2$ and squared error to $b = 0$.

The standard approach to forecast evaluation applies these scoring rules unconditionally. Given a sample of $n$ forecast-realization pairs $\{(\hat{\sigma}^2_t, \sigma^2_t)\}_{t=1}^n$, the unconditional expected loss is estimated by the sample average,
$$
\bar{S} = \frac{1}{n}\sum_{t=1}^{n} S(\hat{\sigma}^2_t, \sigma^2_t),
$$
and models are ranked by their average loss. The Diebold-Mariano test [@diebold1995comparing] provides a formal comparison by testing whether the mean loss differential between two models is zero. This unconditional evaluation is the standard in the volatility forecasting literature, as exemplified by @hansen2005forecast's comparison of 330 volatility models and the widespread use of QLIKE as a ranking criterion.

The implicit assumption underlying unconditional evaluation is that a model that minimizes average loss across all periods is the preferred model for all purposes. As we show in the sections that follow, this assumption can fail when forecasting models are designed to perform differently across regime states.

## 2.2 Conditional Elicitability Under Regime-Switching

We now extend the elicitability framework to regime-conditional settings. Let $G_t \in \mathcal{G}$ denote a regime state at time $t$, where $\mathcal{G}$ is a finite or compact set of possible regime values. In the discrete case, $\mathcal{G} = \{1, 2, \ldots, K\}$ represents $K$ distinct regimes (e.g., low and high volatility states). In the continuous case, $\mathcal{G} = [0,1]$ represents a smooth regime index such as the transition function value in an STR model. For concreteness, we develop the theory for the discrete case and note where the continuous extension requires additional care.

Let $F_{Y|G=g}$ denote the conditional distribution of $Y$ given regime state $G = g$, and let $T_g \equiv T(F_{Y|G=g})$ denote the functional of interest evaluated under this conditional distribution. For volatility forecasting, $T_g = \mathbb{E}[\sigma^2_{t+h} | G_t = g]$ is the conditional variance given regime state $g$.

The regime-conditional expected loss for a forecast $\hat{y}$ under regime $g$ is
$$
L(g; \hat{y}) = \mathbb{E}[S(\hat{y}, Y) \mid G = g].
$$

The key question is whether the scoring function $S$, which is strictly consistent for $T$ under the unconditional distribution $F$, remains strictly consistent for the conditional functional $T_g$ under the conditional distribution $F_{Y|G=g}$. The following proposition establishes that strict consistency is indeed preserved under conditioning.

**Proposition 1** (Conditional consistency of proper scoring rules). Let $S: \mathbb{R} \times \mathbb{R} \to [0, \infty)$ be a strictly consistent scoring function for the functional $T: \mathcal{F} \to \mathbb{R}$ relative to a convex class $\mathcal{F}$. Let $G$ be a random variable taking values in $\mathcal{G}$ such that, for each $g \in \mathcal{G}$, the conditional distribution $F_{Y|G=g}$ belongs to $\mathcal{F}$, and $\mathbb{E}[S(x, Y) \mid G = g]$ exists and is finite for all $x$ in a neighborhood of $T(F_{Y|G=g})$. Then for each fixed $g \in \mathcal{G}$:

(i) $S$ is strictly consistent for $T_g = T(F_{Y|G=g})$ under the conditional distribution $F_{Y|G=g}$, that is,
$$
\mathbb{E}[S(T_g, Y) \mid G = g] < \mathbb{E}[S(x, Y) \mid G = g] \quad \text{for all } x \neq T_g.
$$

(ii) For Bregman divergences, strict consistency implies that $L(g; x) = \mathbb{E}[S(x, Y) \mid G = g]$ is a strictly convex function of $x$ at the minimizer $T_g$, so that $|\hat{y}_A - T_g| < |\hat{y}_B - T_g|$ in Euclidean distance implies $L(g; \hat{y}_A) < L(g; \hat{y}_B)$ for predictions sufficiently close to $T_g$ [@banerjee2005clustering].

*Proof.* The proof follows from the definition of strict consistency applied to the conditional distribution. Fix $g \in \mathcal{G}$ and let $F_g = F_{Y|G=g}$. By assumption, $F_g \in \mathcal{F}$. Since $S$ is strictly consistent for $T$ relative to $\mathcal{F}$, we have, for all $x \neq T(F_g)$,
$$
\mathbb{E}_{F_g}[S(T(F_g), Y)] < \mathbb{E}_{F_g}[S(x, Y)].
$$
But $\mathbb{E}_{F_g}[S(x, Y)] = \mathbb{E}[S(x, Y) \mid G = g]$ by the definition of the conditional distribution. Therefore,
$$
\mathbb{E}[S(T_g, Y) \mid G = g] < \mathbb{E}[S(x, Y) \mid G = g] \quad \text{for all } x \neq T_g,
$$
which establishes part (i). For part (ii), @banerjee2005clustering showed that every Bregman divergence generates a strictly consistent scoring function whose expected loss is a strictly convex function of the prediction. Specifically, for a Bregman divergence $S_\phi(x, y) = \phi(y) - \phi(x) - \phi'(x)(y - x)$ generated by a strictly convex function $\phi$, the conditional expected loss $L(g; x) = \mathbb{E}[S_\phi(x, Y) \mid G = g]$ inherits strict convexity from $\phi$. By strict convexity of $L(g; \cdot)$ at its minimizer $T_g$, for any predictions $\hat{y}_A, \hat{y}_B$ in a sufficiently small neighborhood of $T_g$, the ordering $|\hat{y}_A - T_g| < |\hat{y}_B - T_g|$ implies $L(g; \hat{y}_A) < L(g; \hat{y}_B)$. This local monotonicity in Euclidean distance follows from the positive definiteness of the Hessian of $L(g; \cdot)$ at $T_g$, which equals $\phi''(T_g) > 0$ by the strict convexity of $\phi$. $\square$

The content of Proposition 1 is that conditioning on regime state does not destroy the propriety of scoring rules, provided the conditional distributions remain within the class $\mathcal{F}$ over which consistency was established. This condition is mild: if $\mathcal{F}$ is the class of distributions with finite second moments, then any regime-conditional distribution inheriting finite second moments satisfies the requirement.

The proposition has an important consequence for model evaluation. If we wish to compare two volatility forecasting models within a specific regime, we can use the same scoring function (QLIKE, squared error, or any other strictly consistent score) applied to the subsample of observations falling in that regime, and the resulting ranking will be a valid comparison of the models' regime-conditional performance. There is no need to construct special "regime-aware" scoring functions; standard scoring functions retain their desirable properties when conditioned on regime state.

However, the fact that conditional rankings are valid within each regime does not imply that they agree with unconditional rankings. The unconditional expected loss is a weighted average of conditional expected losses,
$$
\mathbb{E}[S(\hat{y}, Y)] = \sum_{g \in \mathcal{G}} \pi_g \, \mathbb{E}[S(\hat{y}, Y) \mid G = g],
$$
where $\pi_g = \mathbb{P}(G = g)$ is the unconditional probability of regime $g$. When models differ in their regime-conditional performance and regimes have unequal probabilities, the unconditional ranking is dominated by the model's performance in the most frequent regime. This observation motivates the ranking reversal analysis of the next section.

## 2.3 Ranking Reversals Under Regime Imbalance

We now formalize the conditions under which unconditional loss rankings can reverse relative to regime-conditional rankings. This result provides the central theoretical motivation for regime-conditional evaluation.

We note at the outset that the algebraic content of the results in this section is straightforward; their value lies in formalizing the conditions under which unconditional evaluation fails for regime-switching models and in quantifying the relationship between regime imbalance and ranking reversal.

Consider two competing forecasting models, $A$ and $B$, and a strictly consistent scoring function $S$. Define the regime-conditional expected loss of model $m \in \{A, B\}$ under regime $g$ as
$$
L_m(g) = \mathbb{E}[S(\hat{y}_m, Y) \mid G = g],
$$
and the regime-conditional loss differential as
$$
d(g) = L_A(g) - L_B(g).
$$

The unconditional loss differential is
$$
\bar{d} = \mathbb{E}[S(\hat{y}_A, Y)] - \mathbb{E}[S(\hat{y}_B, Y)] = \sum_{g \in \mathcal{G}} \pi_g \, d(g).
$$

The unconditional ranking agrees with the conditional ranking in every regime if and only if $d(g)$ has the same sign for all $g \in \mathcal{G}$. When $d(g)$ changes sign across regimes---model $A$ is better in some regimes and worse in others---the unconditional ranking depends on the regime probabilities $\pi_g$, and ranking reversals become possible.

**Definition 3** (Ranking reversal). A ranking reversal occurs when model $A$ has lower unconditional loss than model $B$ ($\bar{d} < 0$) but model $B$ has lower conditional loss in at least one regime $g^*$ ($d(g^*) > 0$). A ranking reversal *with respect to a pre-specified regime* $g^*$ occurs when $\bar{d} < 0$ but $d(g^*) > 0$ for a regime $g^*$ designated a priori as the regime of greatest economic interest (e.g., the high-volatility stress regime for risk management applications).

We now establish the conditions under which ranking reversals occur.

**Theorem 1** (Ranking reversal under regime imbalance). Consider a two-regime setting with $\mathcal{G} = \{L, H\}$ (low and high volatility), regime probabilities $\pi_L$ and $\pi_H = 1 - \pi_L$, and a strictly consistent scoring function $S$. Let models $A$ and $B$ produce forecasts with regime-conditional loss differentials $d(L) = L_A(L) - L_B(L)$ and $d(H) = L_A(H) - L_B(H)$.

Suppose model $A$ is a regime-switching model that is better in the high-volatility regime but worse in the low-volatility regime, while model $B$ is a simpler model that is uniformly adequate:
$$
d(L) > 0 \quad \text{and} \quad d(H) < 0.
$$

Then the following hold:

(i) Model $B$ is ranked as the unconditional winner (i.e., $\bar{d} > 0$, so $A$ loses unconditionally) if and only if
$$
\pi_L \, d(L) + \pi_H \, d(H) > 0 \quad \iff \quad \frac{\pi_L}{\pi_H} > \frac{-d(H)}{d(L)} = \frac{|d(H)|}{d(L)}.
$$

(ii) A ranking reversal occurs---$B$ wins unconditionally while $A$ wins in the high-volatility regime---whenever the regime imbalance ratio satisfies
$$
\frac{\pi_L}{\pi_H} > \frac{|d(H)|}{d(L)}.
$$

(iii) For the empirically relevant case where $d(L) \approx |d(H)|$ (the regime-switching model's gain in stress is comparable in magnitude to its cost in calm), a ranking reversal occurs whenever $\pi_L > \pi_H$, that is, whenever the low-volatility regime is more frequent than the high-volatility regime.

(iv) *Finite-sample probability bound.* Let $\hat{d}(g)$ denote the sample loss differential in regime $g$, with $\text{Var}(\hat{d}(g)) = \sigma^2_g / n_g$ where $n_g$ is the number of observations in regime $g$. If the population satisfies $d(H) < 0 < d(L)$ and $\bar{d} > 0$ (i.e., B wins unconditionally), then for any sample of size $n$ the probability that the sample unconditional ranking correctly reflects the unconditional population ranking satisfies
$$
\mathbb{P}(\hat{\bar{d}} > 0) \geq 1 - \exp\left(-\frac{n \bar{d}^2}{2(\pi_L \sigma_L^2 + \pi_H \sigma_H^2)}\right),
$$
under sub-Gaussian tail assumptions on the loss differential process. Meanwhile, the probability that the sample detects the reversal in the stress regime (i.e., that $\hat{d}(H) < 0$) satisfies the analogous bound with $n_H = n\pi_H$ replacing $n$. Since $n_H \ll n$ when $\pi_H \ll 1$, substantially larger samples are needed to detect conditional superiority than unconditional superiority.

*Proof.* The unconditional loss differential decomposes as
$$
\bar{d} = \pi_L \, d(L) + \pi_H \, d(H).
$$

Since $d(L) > 0$ (model $A$ is worse in calm) and $d(H) < 0$ (model $A$ is better in stress), the sign of $\bar{d}$ depends on the relative magnitudes of the two terms. Setting $\bar{d} > 0$:
$$
\pi_L \, d(L) + \pi_H \, d(H) > 0
$$
$$
\pi_L \, d(L) > -\pi_H \, d(H) = \pi_H \, |d(H)|
$$
$$
\frac{\pi_L}{\pi_H} > \frac{|d(H)|}{d(L)}.
$$

This establishes part (i). Part (ii) is a direct restatement: the condition $\bar{d} > 0$ (B wins unconditionally) combined with $d(H) < 0$ (A wins in high regime) constitutes a ranking reversal by Definition 3.

For part (iii), when $d(L) \approx |d(H)|$, the right-hand side of the inequality is approximately one. Therefore the condition reduces to $\pi_L / \pi_H > 1$, or equivalently $\pi_L > 1/2$. Since in financial markets the low-volatility regime typically occupies 70--85\% of observations [@hamilton1989new; @guidolin2007asset], the condition $\pi_L > \pi_H$ is almost always satisfied.

For part (iv), the sample unconditional loss differential $\hat{\bar{d}} = n^{-1}\sum_{t=1}^n \Delta L_t$ is a weighted sum of within-regime sample means. Under sub-Gaussian assumptions, Hoeffding's inequality applied to the loss differential process gives $\mathbb{P}(\hat{\bar{d}} \leq 0) \leq \exp(-n\bar{d}^2 / (2V))$ where $V = \pi_L \sigma_L^2 + \pi_H \sigma_H^2$ is the unconditional variance of the loss differential. The analogous bound for $\hat{d}(H)$ replaces $n$ with the effective sample size $n_H = n\pi_H$. The ratio $n / n_H = 1/\pi_H$ quantifies the additional sample size needed for conditional inference. $\square$

Theorem 1 reveals a systematic bias in unconditional forecast evaluation: when market regimes are imbalanced, as they invariably are in financial data, unconditional loss functions overweight performance during calm periods. A regime-switching model that sacrifices a modest amount of accuracy during calm periods to achieve substantially better forecasts during stress may be rejected by unconditional evaluation simply because calm periods dominate the sample. Part (iv) quantifies this effect: detecting conditional superiority in the stress regime requires a sample roughly $1/\pi_H$ times larger than detecting unconditional differences, which for $\pi_H \approx 0.2$ implies a five-fold increase in sample size requirements.

The result extends naturally to the $K$-regime case. Let $\mathcal{G} = \{1, \ldots, K\}$ with probabilities $\pi_1, \ldots, \pi_K$.

**Corollary 1** (Multi-regime ranking reversal). In the $K$-regime setting, a ranking reversal with respect to regime $g^*$ occurs whenever
$$
\sum_{g \neq g^*} \pi_g \, d(g) > \pi_{g^*} \, |d(g^*)|.
$$
That is, the cumulative disadvantage of model $A$ in all other regimes, weighted by their probabilities, outweighs model $A$'s advantage in the regime of interest. This condition is increasingly easy to satisfy as $\pi_{g^*}$ decreases, implying that ranking reversals are most likely for rare but economically important regimes.

The practical implication of Theorem 1 and its corollary is that unconditional QLIKE rankings carry a built-in bias against regime-switching models in precisely the setting where such models are most needed: when the regime of interest (crisis, high volatility, systemic stress) is rare relative to the dominant regime (calm, normal markets). This provides a theoretical rationale for regime-conditional evaluation.

## 2.4 Conditions for Agreement Between Unconditional and Conditional Rankings

While Section 2.3 established conditions for ranking reversal, it is equally important to characterize conditions under which unconditional evaluation is reliable. This section provides sufficient conditions for unconditional and conditional rankings to agree across all regimes, thereby identifying settings where standard evaluation remains valid.

**Theorem 2** (Conditions for ranking agreement). Let $S$ be a strictly consistent scoring function, $\mathcal{G} = \{1, \ldots, K\}$ a finite set of regimes with positive probabilities $\pi_g > 0$ for all $g$, and $d(g) = L_A(g) - L_B(g)$ the regime-conditional loss differential between models $A$ and $B$. Then the unconditional ranking agrees with the conditional ranking in every regime under any of the following sufficient conditions:

(C1) *Uniform dominance:* $d(g) \leq 0$ for all $g \in \mathcal{G}$ (or $d(g) \geq 0$ for all $g$). That is, one model is weakly superior in every regime.

(C2) *Balanced regimes with bounded differentials:* When regime frequencies are balanced ($\pi_g = 1/K$ for all $g$), the unconditional differential equals the unweighted average of conditional differentials:
$$
\bar{d} = \frac{1}{K}\sum_{g=1}^K d(g).
$$
If additionally the magnitude $|d(g)|$ is bounded by some constant $M$ for all $g$, then the unconditional ranking agrees with the modal conditional ranking whenever a strict majority of regimes favor the same model. Specifically, if $|\{g : d(g) < 0\}| > K/2$ and $|d(g)| \leq M$ for all $g$, then $\bar{d} < 0$.

(C3) *Proportional loss differentials:* There exists a constant $\lambda > 0$ such that $|d(g)| = \lambda$ for all $g$, and the set $\{g : d(g) < 0\}$ has total probability exceeding $1/2$. That is, when loss differentials are equal in magnitude across regimes, the unconditional winner is determined by whether the model wins in more than half the probability mass.

*Proof.* We prove each condition separately.

*Condition (C1).* If $d(g) \leq 0$ for all $g$, then
$$
\bar{d} = \sum_{g \in \mathcal{G}} \pi_g \, d(g) \leq 0
$$
since each term is non-positive and $\pi_g > 0$. Moreover, if $d(g) < 0$ for at least one $g$ with $\pi_g > 0$, then $\bar{d} < 0$ strictly. In this case, model $A$ wins both unconditionally and in every regime, so no ranking reversal can occur. The symmetric argument applies when $d(g) \geq 0$ for all $g$.

*Condition (C2).* When $\pi_g = 1/K$ for all $g$,
$$
\bar{d} = \frac{1}{K}\sum_{g=1}^K d(g).
$$
Suppose a strict majority of regimes favor model $A$: let $|\{g: d(g) < 0\}| = J > K/2$. When $|d(g)| \leq M$ for all $g$, each term in the sum is bounded below by $-M$ and above by $M$. The sum over the $J$ favorable regimes contributes at most $-J \cdot \epsilon$ for some $\epsilon > 0$ (since each such $d(g) < 0$), while the sum over the remaining $K - J$ unfavorable regimes contributes at most $(K - J) \cdot M$. However, unlike the equal-magnitude case of (C3), the average of the differentials need not have the same sign as the majority: when magnitudes differ, a minority of regimes with large differentials can outweigh a majority with small differentials. The boundedness condition $|d(g)| \leq M$ limits but does not eliminate this possibility. Full agreement between the unconditional ranking and the modal conditional ranking therefore requires either equal magnitudes across regimes (reducing to (C3)) or the stronger condition that $\sum_{g=1}^K d(g) \neq 0$ with the appropriate sign. The condition $\sum_{g=1}^K d(g) \neq 0$ ensures a determinate ranking.

*Condition (C3).* When $|d(g)| = \lambda$ for all $g$, define $\mathcal{G}^+ = \{g : d(g) > 0\}$ and $\mathcal{G}^- = \{g : d(g) < 0\}$. Then
$$
\bar{d} = \lambda \sum_{g \in \mathcal{G}^+} \pi_g - \lambda \sum_{g \in \mathcal{G}^-} \pi_g = \lambda\left(\pi(\mathcal{G}^+) - \pi(\mathcal{G}^-)\right),
$$
where $\pi(\mathcal{G}^{\pm}) = \sum_{g \in \mathcal{G}^{\pm}} \pi_g$. If $\pi(\mathcal{G}^-) > 1/2$, then $\bar{d} < 0$ and the unconditional ranking agrees with the conditional ranking for the majority of the probability mass. In this case, $A$ wins unconditionally and wins in regimes carrying more than half the probability. $\square$

Condition (C1) is the most practically informative: unconditional evaluation is reliable when one model dominates across all regimes. This corresponds to the case where a model is genuinely and uniformly better, not merely better on average. In the volatility forecasting context, (C1) holds when a model improves upon the benchmark in both calm and stress periods. A model that achieves this uniform dominance should be preferred by any reasonable evaluation criterion, unconditional or conditional.

Condition (C2) highlights that the problem identified in Theorem 1 is fundamentally one of regime imbalance, not of improper scoring. When regimes are balanced, the unconditional average is a fair summary of average behavior, though it does not preclude reversals in individual regimes when magnitudes differ. This condition is almost never satisfied in financial data, where calm periods vastly outnumber stress episodes, but it serves as a theoretical benchmark.

Condition (C3) addresses the special case of symmetric gains and losses across regimes. When the absolute improvement in one regime equals the absolute deterioration in another, the unconditional ranking is determined by which regimes carry more probability mass. This condition formalizes the intuition that regime-switching models face an uphill battle in unconditional evaluation: to win overall, they must improve in regimes that collectively exceed 50\% of the sample.

**Remark 1** (Diagnostic for ranking reliability). The quantity
$$
R = \frac{\max_g d(g) - \min_g d(g)}{|\bar{d}|}
$$
provides a descriptive diagnostic for the severity of regime heterogeneity in loss differentials relative to the overall signal. When $R$ is large (loss differentials vary substantially across regimes relative to their average), unconditional rankings are fragile and may not reflect conditional performance. When $R$ is small (loss differentials are similar across regimes), unconditional rankings are robust. This ratio serves as a practical heuristic for whether regime-conditional evaluation is warranted. We note that $R$ is a sample-dependent statistic whose sampling distribution depends on the joint distribution of loss differentials across regimes. We investigate the sampling properties of $R$ through simulation in Section 3, where we calibrate critical values under various data-generating processes.

## 2.5 Regime-Conditional Model Confidence Set

The Model Confidence Set (MCS) procedure of @hansen2011model provides a systematic approach to model selection that accounts for estimation uncertainty. The MCS identifies a set $\hat{\mathcal{M}}^*_\alpha$ of models that are not significantly outperformed at level $\alpha$ by any other model in the comparison set. This section extends the MCS to regime-conditional settings.

Let $\mathcal{M} = \{1, \ldots, M\}$ denote a set of competing forecasting models. For models $i, j \in \mathcal{M}$, define the regime-conditional loss differential process
$$
d_{ij,t}(g) = S(\hat{y}_{i,t}, y_t) - S(\hat{y}_{j,t}, y_t), \quad \text{for } t \text{ such that } G_t = g,
$$
and the population regime-conditional mean differential
$$
\mu_{ij}(g) = \mathbb{E}[d_{ij,t}(g)] = L_i(g) - L_j(g).
$$

The regime-conditional MCS is the set of models that are not significantly outperformed in regime $g$.

**Definition 4** (Regime-conditional Model Confidence Set). For regime $g \in \mathcal{G}$ and significance level $\alpha \in (0,1)$, the regime-conditional superior set of models is
$$
\mathcal{M}^*_g = \{i \in \mathcal{M} : \mu_{ij}(g) \leq 0 \text{ for all } j \in \mathcal{M}\},
$$
consisting of models whose conditional expected loss in regime $g$ is not exceeded by any other model. The regime-conditional Model Confidence Set $\hat{\mathcal{M}}^*_{g,\alpha}$ is a data-dependent set constructed to satisfy
$$
\lim_{n \to \infty} \mathbb{P}\left(\mathcal{M}^*_g \subseteq \hat{\mathcal{M}}^*_{g,\alpha}\right) \geq 1 - \alpha.
$$

The construction of $\hat{\mathcal{M}}^*_{g,\alpha}$ follows the sequential elimination procedure of @hansen2011model, applied to the subsample of observations falling in regime $g$. Define $n_g = \sum_{t=1}^n \mathbf{1}(G_t = g)$ as the number of observations in regime $g$, and let $\bar{d}_{ij}(g) = n_g^{-1} \sum_{t: G_t = g} d_{ij,t}(g)$ be the sample mean loss differential within regime $g$.

The equivalence hypothesis for regime $g$ is
$$
H_{0,\mathcal{M},g}: \mu_{ij}(g) = 0 \quad \text{for all } i, j \in \mathcal{M}.
$$

We test this using the range statistic adapted to the regime-conditional setting:
$$
T_{R,g} = \max_{i,j \in \mathcal{M}} \frac{|\bar{d}_{ij}(g)|}{\sqrt{\widehat{\text{Var}}(\bar{d}_{ij}(g))}},
$$
where $\widehat{\text{Var}}(\bar{d}_{ij}(g))$ is a HAC-consistent variance estimator applied to the regime-$g$ subsample. The sequential elimination proceeds as in the standard MCS: if the null of equal conditional predictive ability in regime $g$ is rejected, the worst-performing model (the model $i$ that maximizes $\bar{d}_{i \cdot}(g) = (M-1)^{-1} \sum_{j \neq i} \bar{d}_{ij}(g)$) is removed, and the test is repeated on the reduced set.

**Proposition 2** (Validity of regime-conditional MCS). Assume the following conditions hold:

(A1) The joint process $\{(Y_t, G_t, \hat{y}_{1,t}, \ldots, \hat{y}_{M,t})\}$ is strictly stationary and $\alpha$-mixing with mixing coefficients satisfying $\sum_{k=1}^\infty \alpha(k)^{\delta/(2+\delta)} < \infty$ for some $\delta > 0$.

(A2) The effective sample size $n_g \to \infty$ as $n \to \infty$, which requires $\pi_g > 0$.

(A3) A functional central limit theorem applies to $n_g^{-1/2} \sum_{t: G_t = g} (d_{ij,t}(g) - \mu_{ij}(g))$.

Then the regime-conditional MCS $\hat{\mathcal{M}}^*_{g,\alpha}$ has correct asymptotic coverage:
$$
\lim_{n \to \infty} \mathbb{P}\left(\mathcal{M}^*_g \subseteq \hat{\mathcal{M}}^*_{g,\alpha}\right) \geq 1 - \alpha.
$$

*Proof.* The proof proceeds by verifying that the conditions of Theorem 1 in @hansen2011model are satisfied for the regime-conditional subsample. We first address the mixing properties of the regime-conditional subprocess.

Under (A1), the regime-conditional loss differential process $\{d_{ij,t}(g)\}_{t \in \mathcal{T}_g}$, where $\mathcal{T}_g = \{t : G_t = g\}$, inherits mixing properties from the full process. Specifically, the subsequence $\{X_{t_j}\}$ indexed by the regime-$g$ times $t_1 < t_2 < \cdots$ preserves $\alpha$-mixing because the regime indicator process $\{G_t\}$ is itself stationary and mixing under (A1). This follows from a general result on mixing subsequences: if $\{X_t\}$ is $\alpha$-mixing and $\{t_j\}$ is a subsequence determined by a stationary, mixing selection process, then the subsequence $\{X_{t_j}\}$ preserves $\alpha$-mixing with mixing coefficients dominated by those of the original process [@bradley2007introduction, Theorem 3.49]. The key requirement is that the selection mechanism (here, $G_t = g$) does not introduce long-range dependence, which is guaranteed by the mixing assumption on the joint process in (A1).

Under (A2), the regime-conditional subsample grows proportionally with $n$ at rate $\pi_g n$, ensuring a divergent effective sample size.

Under (A3), the regime-conditional sample mean $\bar{d}_{ij}(g)$ satisfies a central limit theorem. This CLT for the subsampled process holds under the inherited mixing conditions by applying the CLT for $\alpha$-mixing sequences [@ibragimov1962some; @merlevede2011bernstein]. The asymptotic variance is consistently estimated by the HAC estimator applied to the regime-$g$ subsample.

Given these conditions, the equivalence test based on $T_{R,g}$ has correct asymptotic size, and the sequential elimination procedure controls the familywise error rate at level $\alpha$ by the same step-down argument as in @hansen2011model. The critical values are obtained by block bootstrap resampling within the regime-$g$ subsample, which consistently estimates the null distribution under (A1)--(A3). $\square$

The regime-conditional MCS framework provides several diagnostic insights not available from unconditional MCS analysis. First, a model may belong to $\hat{\mathcal{M}}^*_{L,\alpha}$ (the set of models not rejected in the calm regime) but not to $\hat{\mathcal{M}}^*_{H,\alpha}$ (rejected in the stress regime), revealing it as a "calm-period specialist." Second, a model in $\hat{\mathcal{M}}^*_{H,\alpha}$ but not $\hat{\mathcal{M}}^*_{L,\alpha}$ is a "stress specialist." Third, a model in the intersection $\hat{\mathcal{M}}^*_{L,\alpha} \cap \hat{\mathcal{M}}^*_{H,\alpha}$ is an "all-weather" model. This taxonomy is informative for practitioners whose loss functions may weight stress-period performance more heavily than calm-period performance.

A practical consideration is the effective sample size within each regime. Since stress periods are rare, $n_H$ may be small even when $n$ is large, reducing the power of the MCS procedure in the high-volatility regime. This is an unavoidable feature of the data---the same rarity that causes unconditional metrics to underweight stress performance also limits the precision of stress-conditional inference. We address this limitation empirically by reporting both asymptotic and bootstrap $p$-values and by conducting power analysis.

## 2.6 Extension to Joint VaR/ES Elicitability Under Regime Conditioning

The Fissler-Ziegel framework [@fissler2016higher] established that while Value-at-Risk (VaR) is elicitable on its own and Expected Shortfall (ES) is not, the pair $(\text{VaR}_\alpha, \text{ES}_\alpha)$ is jointly elicitable. Specifically, there exists a strictly consistent scoring function for the bivariate functional $T(F) = (\text{VaR}_\alpha(F), \text{ES}_\alpha(F))$. The canonical representative of this class, known as the FZ0 loss, is
$$
S_{\text{FZ0}}(\hat{v}, \hat{e}, y) = \frac{1}{\hat{e}} \left(\hat{v} - y + \frac{(y - \hat{v})^+}{\alpha}\right) - \frac{1}{\hat{e}} + \log(-\hat{e}),
$$
where $\hat{v}$ and $\hat{e}$ are the VaR and ES forecasts, $y$ is the realized return, $\alpha$ is the probability level, $(x)^+ = \max(x, 0)$, and $\hat{e} < 0$ denotes the ES at level $\alpha$ for the left tail (i.e., $\hat{e} = \mathbb{E}[Y \mid Y \leq \hat{v}]$ under the forecast distribution, which is negative for loss quantiles).

We extend the regime-conditioning framework to the FZ0 loss. The regime-conditional risk measures are defined as
$$
\text{VaR}_{\alpha,g} = T_1(F_{Y|G=g}), \quad \text{ES}_{\alpha,g} = T_2(F_{Y|G=g}),
$$
where $T_1$ and $T_2$ extract the VaR and ES from the regime-conditional return distribution.

**Proposition 3** (Regime-conditional joint elicitability of VaR and ES). Let $S_{\text{FZ0}}$ be the Fissler-Ziegel scoring function that is strictly consistent for $(\text{VaR}_\alpha, \text{ES}_\alpha)$ relative to a class $\mathcal{F}$ of distributions with finite first moments. If $F_{Y|G=g} \in \mathcal{F}$ for each $g \in \mathcal{G}$, then $S_{\text{FZ0}}$ is strictly consistent for $(\text{VaR}_{\alpha,g}, \text{ES}_{\alpha,g})$ under $F_{Y|G=g}$.

That is, the regime-conditional expected FZ0 loss,
$$
L_{\text{FZ0}}(g; \hat{v}, \hat{e}) = \mathbb{E}\left[S_{\text{FZ0}}(\hat{v}, \hat{e}, Y) \mid G = g\right],
$$
is uniquely minimized at $(\hat{v}, \hat{e}) = (\text{VaR}_{\alpha,g}, \text{ES}_{\alpha,g})$.

*Proof.* The argument is identical in structure to Proposition 1. The joint elicitability result of @fissler2016higher establishes strict consistency of $S_{\text{FZ0}}$ for all distributions in $\mathcal{F}$. Since $F_{Y|G=g} \in \mathcal{F}$ by assumption, the conditional expected score is minimized at the conditional risk measures. The multi-dimensional nature of the functional does not affect the conditioning argument: the minimization over the pair $(\hat{v}, \hat{e})$ under the conditional distribution proceeds exactly as under the unconditional distribution. $\square$

The practical import of Proposition 3 for risk management is substantial. A regime-switching model that calibrates VaR and ES accurately during stress but poorly during calm periods would, under unconditional FZ0 evaluation, be penalized for its calm-period miscalibration. Since calm periods dominate the sample, this model may rank below a simpler model that provides mediocre but uniform calibration. Yet from a risk management perspective, stress-period calibration is far more consequential: VaR violations during calm markets are painful but manageable, while VaR violations during stress can trigger systemic consequences through margin calls, forced liquidation, and regulatory breaches.

The ranking reversal result of Theorem 1 applies directly to the FZ0 loss. Replacing $S$ with $S_{\text{FZ0}}$ and defining $d_{\text{FZ0}}(g)$ as the regime-conditional FZ0 loss differential, the conditions for ranking reversal are identical. A risk management application of regime-conditional evaluation would compare models using $L_{\text{FZ0}}(H; \hat{v}, \hat{e})$---the expected FZ0 loss during stress---as the primary criterion, with unconditional FZ0 serving as a secondary diagnostic.

**Corollary 2** (Regime-conditional risk evaluation). The ranking reversal conditions of Theorem 1, applied to $S_{\text{FZ0}}$, imply that unconditional backtesting of VaR/ES models systematically favors models that calibrate well during calm periods. For regulatory applications where stress-period performance is paramount, regime-conditional FZ0 evaluation provides a more relevant ranking.

## 2.7 Connection to Giacomini-White Conditional Predictive Ability

The conditional predictive ability (CPA) framework of @giacomini2006tests provides a general approach to testing whether two forecasting models have equal predictive accuracy conditional on an information set $\mathcal{H}_t$ available at time $t$. Their test statistic is based on the moment condition
$$
\mathbb{E}[\Delta L_t \cdot h_t] = 0,
$$
where $\Delta L_t = S(\hat{y}_{A,t}, y_t) - S(\hat{y}_{B,t}, y_t)$ is the loss differential and $h_t$ is a vector of test functions measurable with respect to $\mathcal{H}_t$. When $h_t = \mathbf{1}(G_t = g)$ for some regime indicator $G_t$, the test reduces to comparing conditional expected losses in regime $g$, which is precisely the regime-conditional comparison developed in Sections 2.2--2.4.

Our framework differs from @giacomini2006tests in two important respects. First, the Giacomini-White test conditions on observable instruments $h_t$ that are in the forecaster's information set. When the regime variable $G_t$ is observable (e.g., an indicator based on the VIX exceeding a threshold), the regime-conditional test is a direct application of their framework with $h_t = \mathbf{1}(G_t = g)$. In this case, the Giacomini-White test and our regime-conditional comparison are equivalent, and the asymptotic theory follows directly from their results.

Second, and more substantively, our setting involves conditioning on a *latent* regime state that is estimated from the data. Let $G_t^*$ denote the true (latent) regime state and $\hat{G}_t$ denote its estimate (e.g., from a VRNN or a Markov-switching filter). The regime-conditional loss using the estimated regime is
$$
\hat{L}_m(g) = \mathbb{E}[S(\hat{y}_m, Y) \mid \hat{G}_t = g],
$$
which differs from the target $L_m(g) = \mathbb{E}[S(\hat{y}_m, Y) \mid G_t^* = g]$ due to regime estimation error. This introduces an additional source of uncertainty not present in the Giacomini-White framework, which treats the conditioning variables as known.

**Proposition 4** (Asymptotic validity under consistent regime estimation). Let $\hat{G}_t$ be an estimate of the latent regime state $G_t^*$ satisfying the following conditions:

(B1) *Consistency:* $\mathbb{P}(\hat{G}_t \neq G_t^*) \to 0$ as the estimation sample grows.

(B2) *Uniform convergence:* $\sup_{g \in \mathcal{G}} |\hat{\pi}_g - \pi_g| \xrightarrow{p} 0$, where $\hat{\pi}_g = n^{-1}\sum_{t=1}^n \mathbf{1}(\hat{G}_t = g)$.

(B3) *Asymptotic negligibility of misclassification:* $n^{1/2} (M_n / n) \to 0$ in probability, where $M_n = \sum_{t=1}^n \mathbf{1}(\hat{G}_t \neq G_t^*)$ is the total misclassification count.

Then the regime-conditional loss estimate based on $\hat{G}_t$,
$$
\hat{L}_m(g) = \frac{1}{\hat{n}_g} \sum_{t: \hat{G}_t = g} S(\hat{y}_{m,t}, y_t),
$$
is consistent for $L_m(g)$, and the regime-conditional Giacomini-White test statistic using $h_t = \mathbf{1}(\hat{G}_t = g)$ has the same asymptotic distribution as the infeasible test using $h_t = \mathbf{1}(G_t^* = g)$.

*Proof.* We decompose the error into a classification component and an estimation component. Write $\hat{n}_g = \sum_{t=1}^n \mathbf{1}(\hat{G}_t = g)$ and $n_g^* = \sum_{t=1}^n \mathbf{1}(G_t^* = g)$. Consider the decomposition
$$
\hat{L}_m(g) - L_m(g) = \underbrace{\left(\hat{L}_m(g) - \tilde{L}_m(g)\right)}_{\text{classification error}} + \underbrace{\left(\tilde{L}_m(g) - L_m(g)\right)}_{\text{estimation error}},
$$
where $\tilde{L}_m(g) = (n_g^*)^{-1} \sum_{t: G_t^* = g} S(\hat{y}_{m,t}, y_t)$ is the infeasible regime-conditional loss using true regime labels.

The estimation error term converges to zero in probability by the law of large numbers applied to the stationary process $\{S(\hat{y}_{m,t}, y_t) \cdot \mathbf{1}(G_t^* = g)\}$, under the stationarity and mixing conditions assumed throughout.

For the classification error, the total misclassification count is $M_n = \sum_{t=1}^n \mathbf{1}(\hat{G}_t \neq G_t^*)$, and each misclassification perturbs the sum by at most $\|S\|_\infty$ (assuming bounded scores; see Remark 2 below for the extension to unbounded scores). Under (B1), $M_n / n \xrightarrow{p} 0$, so
$$
\left|\hat{L}_m(g) - \tilde{L}_m(g)\right| \leq \frac{2 M_n \|S\|_\infty}{\hat{n}_g},
$$
which converges to zero in probability since $\hat{n}_g / n \xrightarrow{p} \pi_g > 0$ by (B2).

For the distributional result, the regime-conditional test statistic is
$$
\hat{t}_g = \frac{\bar{d}(g)}{\sqrt{\widehat{\text{Var}}(\bar{d}(g)) / \hat{n}_g}},
$$
where $\bar{d}(g) = \hat{L}_A(g) - \hat{L}_B(g)$. Under condition (B3), the contribution of misclassified observations to $\sqrt{\hat{n}_g}\,\bar{d}(g)$ is $o_p(1)$, since $M_n = o_p(n^{1/2})$ by (B3) and the scores are bounded. Therefore, $\sqrt{\hat{n}_g}\,\bar{d}(g)$ has the same limiting distribution as the infeasible statistic $\sqrt{n_g^*}\,\tilde{d}(g)$, which is asymptotically normal under the mixing and CLT conditions of (A1)--(A3) from Proposition 2. $\square$

It is important to distinguish two sources of misclassification in condition (B3). *Estimation-induced misclassification* arises from the finite-sample error in estimating the regime classifier and vanishes asymptotically under (B1). *Population misclassification* arises when the true data-generating process does not admit a perfect classification rule---for instance, when the regime states have overlapping conditional distributions so that the Bayes-optimal classifier has positive error rate. When population misclassification is positive, $M_n / n$ converges to a positive constant, and condition (B3) fails. In this case, the regime-conditional test remains valid as a test of conditional predictive ability given the *estimated* regime $\hat{G}_t$, analogous to the Giacomini-White test with a noisy but informative instrument. The test has correct size for the null hypothesis $\mathbb{E}[\Delta L_t \mid \hat{G}_t = g] = 0$ and retains power against alternatives where models differ conditional on the estimated regime, even though the estimated regime may not perfectly track the latent state.

**Remark 2** (Extension to unbounded scores). The proof of Proposition 4 uses bounded scores to control the classification error term. The QLIKE loss $S_{\text{QLIKE}}(\hat{\sigma}^2, \sigma^2) = \sigma^2/\hat{\sigma}^2 - \log(\sigma^2/\hat{\sigma}^2) - 1$ is unbounded, since it diverges as $\hat{\sigma}^2 \to 0$ or as $\sigma^2 / \hat{\sigma}^2 \to \infty$. The extension to unbounded scores is straightforward under finite moment conditions. Specifically, if $\mathbb{E}[|S(\hat{y}_{m,t}, y_t)|^{2+\delta}] < \infty$ for some $\delta > 0$, then the classification error bound becomes
$$
\left|\hat{L}_m(g) - \tilde{L}_m(g)\right| \leq \frac{2 M_n}{\hat{n}_g} \left(\frac{1}{M_n} \sum_{t \in \mathcal{M}_n} |S(\hat{y}_{m,t}, y_t)|\right),
$$
where $\mathcal{M}_n = \{t : \hat{G}_t \neq G_t^*\}$ is the set of misclassified observations. Under the moment condition, the average score over misclassified observations is $O_p(1)$, and the bound reduces to $O_p(M_n / \hat{n}_g) = o_p(1)$. For the distributional result, the contribution of misclassified observations to $\sqrt{\hat{n}_g}\,\bar{d}(g)$ is controlled by $M_n / n^{1/2}$ times the average score magnitude, which is $o_p(1)$ under (B3) and the finite fourth moment condition $\mathbb{E}[\Delta L_t^4] < \infty$.

For parametric regime models such as Markov-switching, the misclassification probability decreases exponentially in the regime persistence parameter and satisfies (B3) under standard regularity conditions [cf. @douc2004asymptotic]. For the VRNN-based regime classifier used in our empirical application, verifying (B3) analytically requires parametric assumptions on the data-generating process that we are not prepared to impose. We therefore treat (B3) as a maintained assumption and rely on block bootstrap inference over both the regime estimation and forecast comparison stages as the operational approach for valid inference; see the discussion below.

**Generated-regressor uncertainty.** When condition (B3) fails---or more generally, when the regime estimation error makes a non-negligible contribution to the sampling distribution of the test statistic---the regime-conditional comparison inherits a generated-regressor problem. The estimated regime $\hat{G}_t$ enters the test as if it were observed, but it is itself a function of the data, introducing additional variability that is not captured by standard asymptotic standard errors. In this setting, standard errors computed from the second-stage forecast comparison alone may be too small, leading to over-rejection of the null hypothesis of equal conditional predictive ability.

The appropriate inferential strategy is a block bootstrap that resamples over both stages of the procedure: regime estimation and forecast comparison. Specifically, one draws bootstrap samples of contiguous blocks from the original time series, re-estimates the regime classifier on each bootstrap sample, re-computes the regime-conditional loss differentials, and constructs bootstrap confidence intervals for the test statistic. This two-stage bootstrap provides valid inference by implicitly accounting for the joint sampling distribution of the regime estimates and the loss differentials [@goncalves2004bootstrapping]. In our empirical implementation, we employ the circular block bootstrap with block length chosen by the rule of @politis2004automatic, applied to the full two-stage procedure. This approach is conservative in the sense that it captures both regime estimation uncertainty and forecast evaluation uncertainty, at the cost of somewhat wider confidence intervals relative to the infeasible procedure with known regimes.

The connection to @giacomini2006tests also illuminates the relationship between regime-conditional evaluation and unconditional evaluation. In their framework, the unconditional Diebold-Mariano test is a special case where $h_t = 1$ (no conditioning). The regime-conditional test with $h_t = \mathbf{1}(\hat{G}_t = g)$ is a strict refinement: it can detect differences in predictive ability that are masked by the unconditional test when model superiority varies across regimes. The unconditional test can reject the null of equal predictive ability only when the average loss differential is non-zero; the regime-conditional test can reject when the loss differential is non-zero in a specific regime, even if it averages to zero unconditionally.

**Corollary 3** (Refinement of Diebold-Mariano). Let $\bar{d} = \mathbb{E}[\Delta L_t]$ denote the unconditional mean loss differential and $d(g) = \mathbb{E}[\Delta L_t \mid G_t^* = g]$ the regime-conditional mean loss differential. If there exist regimes $g_1, g_2$ with $d(g_1) < 0 < d(g_2)$ and $\sum_g \pi_g d(g) = 0$, then the unconditional Diebold-Mariano test has no power to detect the difference in predictive ability (since $\bar{d} = 0$), but the regime-conditional test rejects the null of equal conditional predictive ability in each regime for which $d(g) \neq 0$, provided the subsample is sufficiently large. Regime-conditional evaluation is therefore a strict refinement of unconditional evaluation in the sense that it can detect differences invisible to the unconditional test.
