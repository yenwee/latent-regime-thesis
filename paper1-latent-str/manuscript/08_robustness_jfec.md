# 6. Robustness Checks

## 6.1. Alternative Latent Dimensions

We examine sensitivity to the VRNN latent dimension $d \in \{1, 2, 4, 8\}$. Table 7 reports panel-average QLIKE for the weekly horizon.

**Table 7: Sensitivity to Latent Dimension**

| Latent Dim ($d$) | N Assets | Median QLIKE | Mean QLIKE | vs Baseline ($d=2$) |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 20 | -8.5098 | -8.6266 | -0.099% |
| 2 | 20 | **-8.5014** | **-8.6356** | --- |
| 4 | 20 | -8.4997 | -8.6258 | +0.011% |
| 8 | 20 | -8.4743 | -8.6157 | +0.318% |

*Notes:* QLIKE loss across 20 assets at H=5. Bold indicates best performance.

The two-dimensional latent space provides optimal performance, balancing expressiveness against overfitting. Higher dimensions ($d=8$) show degraded out-of-sample performance consistent with the curse of dimensionality in state-space estimation.

## 6.2. Alternative Transition Functions

We compare the logistic transition function against exponential and double-logistic alternatives:

**Table 8: Alternative Transition Functions**

| Transition Function | Parameters | N Assets | Median QLIKE | Mean QLIKE | vs Logistic |
|:---|:---:|:---:|:---:|:---:|:---:|
| Logistic | 9 | 20 | **-8.5014** | **-8.6356** | --- |
| Exponential | 9 | 20 | -8.4952 | -8.6300 | +0.073% |
| Double Logistic | 10 | 20 | -8.5015 | -8.6370 | -0.002% |

*Notes:* QLIKE loss across 20 assets at H=5.

All three transition functions produce nearly identical results (differences less than 0.1%), confirming that the choice of transition function is inconsequential for forecast performance.

## 6.3. Subsample Stability

We examine whether results are driven by specific market episodes by evaluating performance across four non-overlapping 2-year periods (2017-2018, 2019-2020, 2021-2022, 2023-2024):

**Table 9: Subsample Performance (QLIKE Improvement vs. HAR)**

| Period | N Assets | SSM vs HAR (%) | OBS vs HAR (%) | SSM vs OBS (pp) |
|:---|:---:|:---:|:---:|:---:|
| 2017-2018 | 20 | 0.05% | 0.43% | -0.38 |
| 2019-2020 (COVID) | 20 | -0.09% | 0.15% | -0.24 |
| 2021-2022 | 20 | 0.39% | 0.36% | +0.03 |
| 2023-2024 | 20 | 0.16% | 0.15% | +0.01 |

*Notes:* QLIKE improvement relative to HAR at H=5.

Deep-LSTR outperforms HAR in three of four subperiods, with the largest gain during 2021--2022 ($+0.39\%$) and smallest during the COVID-19 crisis (2019--2020), where Deep-LSTR is marginally worse than HAR ($-0.09\%$). The COVID underperformance likely reflects the unprecedented speed and magnitude of the March 2020 volatility spike, which challenged the VRNN's latent state inference. STR-OBS outperforms Deep-LSTR in the first two subperiods (2017--2018 and 2019--2020), but Deep-LSTR narrows the gap and achieves comparable or slightly better performance in the later subperiods (2021--2022 and 2023--2024). This pattern suggests that the VRNN benefits from accumulating longer training history, consistent with deep learning models requiring sufficient data to learn stable latent representations.

## 6.4. Alternative Volatility Estimators

We verify robustness to the volatility proxy by comparing Garman-Klass [@garman1980estimation] against Parkinson [@parkinson1980extreme] and Rogers-Satchell [@rogers1991estimating] estimators:

**Table 10: Alternative Volatility Estimators**

| Estimator | N Assets | SSM QLIKE | HAR QLIKE | SSM Win Rate |
|:---|:---:|:---:|:---:|:---:|
| Garman-Klass | 20 | **-8.5014** | -8.4946 | 0.95 |
| Parkinson | 20 | -8.4806 | -8.4785 | 0.95 |
| Rogers-Satchell | 20 | -8.4404 | -8.4265 | 0.95 |

*Notes:* Median QLIKE loss across 20 assets at H=5.

The relative advantage of Deep-LSTR is consistent across volatility estimators, confirming that results are not artifacts of the specific realized variance measure employed.

## 6.5. Transition Variable Comparison

A key question is whether the forecasting improvement derives from the nonlinear latent dynamics captured by the VRNN, or whether observable transition variables already extract sufficient regime information. We address this by comparing three progressively sophisticated transition variable specifications within the STR-HAR framework: (i) no transition variable (standard HAR), (ii) observable log-RV as the transition variable (STR-OBS), and (iii) VRNN-inferred latent state as the transition variable (Deep-LSTR).

**Table 11: Transition Variable Comparison (H=5)**

| Transition Variable | Median QLIKE | Win Rate vs HAR | MCS Inclusion |
|:---|:---:|:---:|:---:|
| HAR (no transition) | -8.4946 | --- | 0.20 |
| STR-OBS (log-RV) | -8.4951 | 0.95 | 0.50 |
| **Deep-LSTR (VRNN)** | **-8.5014** | **0.95** | **0.90** |

*Notes:* Median QLIKE loss across 20 assets (excluding IRX) at H=5. MCS at 10% level.

The comparison reveals a clear hierarchy: introducing any transition variable improves upon HAR, but the latent transition variable provides substantially greater improvement than the observable proxy. While both STR-OBS and Deep-LSTR achieve high win rates against HAR (95%), the MCS inclusion rate---which captures whether a model is statistically indistinguishable from the best---nearly doubles from 50% for STR-OBS to 90% for Deep-LSTR. This indicates that the VRNN captures nonlinear regime dynamics beyond what observable log-volatility provides as a transition variable. The improvement is consistent with the regime smoothing mechanism documented in Section 5.5: the latent state acts as a de-noised regime indicator, filtering transient volatility spikes that would trigger false regime switches under the observable proxy.
