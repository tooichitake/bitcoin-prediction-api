"""Bitcoin price prediction package"""

from bitcoin.features import (
    BitcoinDataProcessor,
    inverse_transform_predictions,
    OPTIMAL_FEATURES_30
)
from bitcoin.plots import (
    plot_candlestick,
    plot_target_distribution,
    plot_features_correlation,
    plot_regression_results,
    analyze_optimal_feature_count,
    plot_optimal_feature_count
)
from bitcoin.modeling.train import (
    TimesNetRegressor,
    save_model,
    load_model
)
from bitcoin.modeling.predict import (
    evaluate_regression
)

__all__ = [
    'BitcoinDataProcessor',
    'inverse_transform_predictions',
    'OPTIMAL_FEATURES_30',
    'plot_candlestick',
    'plot_target_distribution',
    'plot_features_correlation',
    'plot_regression_results',
    'analyze_optimal_feature_count',
    'plot_optimal_feature_count',
    'TimesNetRegressor',
    'save_model',
    'load_model',
    'evaluate_regression'
]
