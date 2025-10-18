"""Model evaluation module"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error


def evaluate_regression(y_true, y_pred, set_name='Test'):
    """
    Unified evaluation for any regression model

    Args:
        y_true: True values (original scale)
        y_pred: Predicted values (original scale)
        set_name: Name for display

    Returns:
        dict: Metrics dictionary with MAE, RMSE, MAPE, R²
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)

    print(f"\n{set_name.upper()} SET PERFORMANCE:")
    print("=" * 60)
    print(f"  MAE:  ${mae:,.2f}")
    print(f"  RMSE: ${rmse:,.2f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R2:   {r2:.4f}")
    print("=" * 60)

    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2,
        'y_true': y_true,
        'y_pred': y_pred
    }
