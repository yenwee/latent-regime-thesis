# Paper 3: Economic Validation of Latent Volatility Regimes
# Validation and interpretation code (no new forecasting models)

# Import from Paper 1
import sys
from pathlib import Path

# Paper 1: Core infrastructure (VRNN, STR-HAR, metrics)
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'paper1-latent-str'))

# Paper 3 modules
from .external_data import (
    fetch_yahoo,
    fetch_fred,
    load_stress_proxies,
    compute_vrp,
    compute_rolling_correlation,
)
from .regime_loader import (
    load_regime_panel,
    load_regime_series,
    get_asset_list,
    align_regime_and_external,
)
