# Paper 2: Cross-Asset Shared Latent Volatility Regimes
# Paper 2-specific extensions (reuses Paper 1's src/ for core functionality)

# Import from Paper 1
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'paper1-latent-str'))

# Paper 2-specific modules (to be implemented)
# from .multivariate_vrnn import MultiVariateVRNN, train_multivariate_ssm
# from .shared_regime import SharedRegimeSTR, fit_shared_regime
# from .regime_benchmarks import PCARegime, VIXRegime, CrossAvgRegime
# from .synchronization import compute_regime_sync, stress_vs_tranquil
# from .partial_pooling import estimate_loadings, heterogeneity_analysis
