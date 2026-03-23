# Deep-LSTR: Deep Latent Smooth-Transition HAR for Volatility Forecasting
# Module initialization

from .utils import set_seeds, load_config, ensure_writable_dir
from .data import (
    flatten_yf_columns,
    get_price_series,
    garman_klass_var,
    download_asset_data,
    prepare_features,
)
from .metrics import qlike, fz0_loss, mse_logv
from .dm_test import dm_test, newey_west_var
from .mcs import bootstrap_mcs
# vrnn intentionally NOT imported here to avoid PyTorch's ~400GB VSIZE on Apple Silicon.
# Import directly: from src.vrnn import DeepSSM, train_deep_ssm
from .str_har import (
    logistic,
    fit_har_ols,
    har_predict,
    fit_str2_window_robust,
    str2_forecast_one,
    str2_in_sample_yhat,
)
from .risk import (
    t_var_es_var1,
    fit_nu_mle_var1,
    risk_series_var_es_dynamic,
    kupiec_test,
)
from .garch import Garch11_t, Egarch11_t, MSGarch2_t

__version__ = "0.1.0"
