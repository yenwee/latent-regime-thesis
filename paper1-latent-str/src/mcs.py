"""
Model Confidence Set (MCS) implementation.
Based on Hansen, Lunde, and Nason (2011).
"""

import numpy as np


def bootstrap_mcs(losses, alpha=0.05, n_boot=1000, block_size=10):
    """
    Simple Block Bootstrap MCS implementation.

    Implements the Model Confidence Set procedure of Hansen, Lunde, and Nason (2011).
    Uses circular block bootstrap for HAC-consistent inference.

    Args:
        losses: DataFrame (T x M) where columns are model names
                and rows are time periods
        alpha: Significance level for MCS (default 0.05)
        n_boot: Number of bootstrap replications (default 1000)
        block_size: Block size for circular block bootstrap (default 10)

    Returns:
        Tuple of:
            - List of model names in MCS (those not rejected)
            - Dict of MCS p-values for all models
    """
    T, M = losses.shape
    model_names = losses.columns.tolist()

    # Generate bootstrap indices (Circular Block Bootstrap)
    bs_indices = []
    for _ in range(n_boot):
        # Random start points
        starts = np.random.randint(0, T, size=int(np.ceil(T / block_size)))
        idx = []
        for s in starts:
            idx.extend(np.arange(s, s + block_size) % T)
        bs_indices.append(idx[:T])

    bs_indices = np.array(bs_indices)  # (n_boot, T)

    # Loop to eliminate models
    inc = model_names.copy()
    removal_order = []
    step_pvals = []

    while len(inc) > 1:
        loss_sub = losses[inc].values
        loss_mean = loss_sub.mean(axis=1, keepdims=True)
        d_i = loss_sub - loss_mean
        d_bar = d_i.mean(axis=0)

        bs_d_i = d_i[bs_indices]  # (B, T, m)
        bs_d_bar = bs_d_i.mean(axis=1)  # (B, m)
        var_d_bar = bs_d_bar.var(axis=0)

        t_stat = d_bar / np.sqrt(var_d_bar + 1e-12)
        TR_obs = np.max(t_stat)

        bs_centered = bs_d_bar - d_bar
        t_stat_b = bs_centered / np.sqrt(var_d_bar + 1e-12)
        TR_b = np.max(t_stat_b, axis=1)

        p = np.mean(TR_b >= TR_obs)
        step_pvals.append(p)

        worst_idx = np.argmax(t_stat)
        removal_order.append(inc[worst_idx])
        inc.pop(worst_idx)

    removal_order.append(inc[0])
    step_pvals.append(1.0)

    # Compute MCS p-values
    # MCS_p[i] = max_{j <= i} step_pvals[j] where i is index in removal_order
    mcs_pvals = {}
    curr = 0.0
    for m, p in zip(removal_order, step_pvals):
        curr = max(curr, p)
        mcs_pvals[m] = curr

    # Return models with p >= alpha
    return [m for m, p in mcs_pvals.items() if p >= alpha], mcs_pvals
