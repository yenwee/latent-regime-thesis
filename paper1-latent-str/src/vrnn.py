"""
Deep State-Space Model (VRNN) for latent regime identification.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DeepSSM(nn.Module):
    """
    Deep State-Space Model with diagonal AR(1) latent dynamics.

    Architecture:
        - GRU encoder for causal inference
        - Diagonal AR(1) latent process with learnable persistence
        - MLP decoder for observation reconstruction

    This implements a Variational Recurrent Neural Network (VRNN)
    with structured latent dynamics appropriate for volatility modeling.
    """

    def __init__(self, p, gru_hidden=16, dec_hidden=32, latent_dim=2):
        """
        Initialize DeepSSM.

        Args:
            p: Input dimension (number of features)
            gru_hidden: GRU hidden state dimension (default 16)
            dec_hidden: Decoder hidden layer dimension (default 32)
            latent_dim: Latent state dimension (default 2)
        """
        super().__init__()
        self.p = p
        self.k = latent_dim

        # GRU encoder
        self.gru = nn.GRU(
            input_size=p, hidden_size=gru_hidden, num_layers=1, batch_first=True
        )
        self.to_m = nn.Linear(gru_hidden, self.k)
        self.to_log_s = nn.Linear(gru_hidden, self.k)

        # MLP decoder
        self.dec = nn.Sequential(
            nn.Linear(self.k, dec_hidden),
            nn.Tanh(),
            nn.Linear(dec_hidden, dec_hidden),
            nn.Tanh(),
            nn.Linear(dec_hidden, p),
        )

        # AR(1) latent dynamics parameters
        self.rho_unconstr = nn.Parameter(torch.zeros(self.k) + 0.3)
        self.log_sigma_eta = nn.Parameter(torch.zeros(self.k) - 2.0)
        self.log_sigma_x = nn.Parameter(torch.zeros(p) - 0.5)

    def rho(self):
        """AR(1) persistence parameter, constrained to (-1, 1)."""
        return torch.tanh(self.rho_unconstr)

    def sigma_eta(self):
        """Latent innovation standard deviation."""
        return torch.exp(self.log_sigma_eta)

    def sigma_x(self):
        """Observation noise standard deviation."""
        return torch.exp(self.log_sigma_x)

    def infer_q(self, X_seq):
        """
        Infer approximate posterior q(z|x) using GRU encoder.

        Args:
            X_seq: Input sequence tensor (T x p)

        Returns:
            Tuple of (mean, std) for approximate posterior
        """
        out, _ = self.gru(X_seq.unsqueeze(0))
        h = out.squeeze(0)
        m = self.to_m(h)
        log_s = self.to_log_s(h).clamp(-8.0, 4.0)
        s = torch.exp(log_s)
        return m, s

    def dec_mean(self, z):
        """Decode latent state to observation mean."""
        return self.dec(z)


def train_deep_ssm(
    X_train,
    X_infer,
    latent_dim=2,
    gru_hidden=16,
    dec_hidden=32,
    lr=2e-3,
    weight_decay=1e-4,
    epochs=600,
    patience=60,
    device=None,
    verbose=False,
):
    """
    Train DeepSSM model using variational inference.

    Maximizes the Evidence Lower Bound (ELBO) with early stopping.

    Args:
        X_train: Training data tensor (T_train x p)
        X_infer: Data for inference (T_infer x p), typically full sample
        latent_dim: Latent state dimension
        gru_hidden: GRU hidden dimension
        dec_hidden: Decoder hidden dimension
        lr: Learning rate
        weight_decay: L2 regularization
        epochs: Maximum training epochs
        patience: Early stopping patience
        device: Torch device (default CPU)
        verbose: Print training progress

    Returns:
        Tuple of (trained model, inferred latent states, final ELBO/T)
    """
    if device is None:
        device = torch.device("cpu")

    model = DeepSSM(
        p=X_train.shape[1],
        gru_hidden=gru_hidden,
        dec_hidden=dec_hidden,
        latent_dim=latent_dim,
    ).to(device)

    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def mc_elbo(X_seq):
        """Monte Carlo ELBO estimate with single sample."""
        m, s = model.infer_q(X_seq)
        z = m + s * torch.randn_like(m)
        X_mean = model.dec_mean(z)

        sigx = model.sigma_x()
        varx = sigx ** 2
        logpx = (
            -0.5 * (np.log(2 * np.pi) + torch.log(varx) + (X_seq - X_mean) ** 2 / varx)
        ).sum()

        rho = model.rho()
        se2 = model.sigma_eta() ** 2

        # Prior on z[0] is N(0, 1)
        logpz = (-0.5 * (np.log(2 * np.pi) + (z[0] ** 2))).sum()
        # AR(1) transitions
        innov = z[1:] - rho * z[:-1]
        logpz += (-0.5 * (np.log(2 * np.pi) + torch.log(se2) + (innov ** 2) / se2)).sum()

        # Entropy of q
        logqz = (
            -0.5 * (np.log(2 * np.pi) + 2 * torch.log(s) + ((z - m) ** 2) / (s ** 2))
        ).sum()

        return logpx + logpz - logqz

    best_obj = -1e18
    best_state = None
    bad = 0
    last_elbo = np.nan

    model.train()
    for epoch in range(1, epochs + 1):
        opt.zero_grad()
        elbo = mc_elbo(X_train)
        loss = -elbo / X_train.shape[0]
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        cur = float(elbo.detach().cpu())
        last_elbo = cur / X_train.shape[0]

        if cur > best_obj + 1e-4:
            best_obj = cur
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            bad = 0
        else:
            bad += 1

        if verbose and (epoch % 100 == 0 or epoch == 1):
            print(
                f"    [SSM epoch {epoch:4d}] ELBO/T={last_elbo:.4f} "
                f"rho_mean={float(model.rho().mean().detach().cpu()):.3f}"
            )

        if bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        m_infer, _ = model.infer_q(X_infer)
    Z_infer = m_infer.detach().cpu().numpy()

    return model, Z_infer, float(last_elbo)


def project_latent_to_scalar(Z_infer, y_target, train_end_idx):
    """
    Project latent states to scalar via supervised linear projection.

    Uses OLS regression on training data to find weights that maximize
    correlation with the target variable (typically log volatility).
    Supports any latent dimension (automatically inferred from Z_infer).

    Args:
        Z_infer: Inferred latent states array (T x latent_dim)
        y_target: Target variable for supervision (T,), e.g., log realized volatility
        train_end_idx: Index marking end of training period (exclusive)

    Returns:
        q: Projected scalar series (T,), standardized and sign-corrected

    Notes:
        - Standardization uses training data statistics only (mean, std)
        - OLS fit: y_train ~ 1 + Z1_train + Z2_train
        - Weights w = [beta_1, beta_2] applied to standardized Z
        - Sign flipped if correlation with training target is negative
    """
    # Extract training portion of latent states
    Z_train = Z_infer[:train_end_idx]

    # Compute standardization statistics from training data only
    Z_mean = Z_train.mean(axis=0, keepdims=True)
    Z_std = Z_train.std(axis=0, keepdims=True)
    Z_std[Z_std < 1e-9] = 1.0  # Prevent division by zero

    # Standardize full Z using training statistics
    Zs = (Z_infer - Z_mean) / Z_std

    # Build design matrix with intercept for training data
    y_train = y_target[:train_end_idx]
    latent_dim = Zs.shape[1]
    Xw = np.column_stack([np.ones(train_end_idx)] + [Zs[:train_end_idx, d] for d in range(latent_dim)])

    # Fit OLS: y_train ~ 1 + Z1 + ... + Zd
    beta_w, *_ = np.linalg.lstsq(Xw, y_train, rcond=None)

    # Extract projection weights (exclude intercept)
    w = beta_w[1:]

    # Project standardized Z to scalar
    q = Zs @ w

    # Ensure positive correlation with target on training data
    if np.corrcoef(q[:train_end_idx], y_train)[0, 1] < 0:
        q = -q

    return q
