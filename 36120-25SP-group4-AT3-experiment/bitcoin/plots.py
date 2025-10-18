"""Bitcoin data visualization functions"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not available. Install with: pip install plotly")

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def plot_candlestick(df: pd.DataFrame, date_col: str = 'timeOpen',
                     n_days: int = None, title: str = "Bitcoin Price"):
    """
    Plot interactive candlestick chart with volume and market cap using Plotly

    Args:
        df: DataFrame with OHLCV and marketCap data
        date_col: Date column name
        n_days: Number of recent days (None = all data)
        title: Chart title
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not installed. Install with: pip install plotly")
        return

    df_plot = df.tail(n_days) if n_days else df.copy()

    # Create subplots: 3 rows
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(title, 'Volume', 'Market Cap')
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df_plot[date_col],
            open=df_plot['open'],
            high=df_plot['high'],
            low=df_plot['low'],
            close=df_plot['close'],
            name='OHLC'
        ),
        row=1, col=1
    )

    # Volume with stronger colors and full opacity (no transparency)
    # Use solid colors for better visibility at all zoom levels
    colors = ['#EF5350' if row['close'] < row['open'] else '#26A69A'
              for _, row in df_plot.iterrows()]

    fig.add_trace(
        go.Bar(
            x=df_plot[date_col],
            y=df_plot['volume'],
            name='Volume',
            marker=dict(
                color=colors,
                line=dict(width=0)
            ),
            showlegend=False
        ),
        row=2, col=1
    )

    # Market Cap
    fig.add_trace(
        go.Scatter(
            x=df_plot[date_col],
            y=df_plot['marketCap'],
            name='Market Cap',
            line=dict(color='#1976D2', width=1.5),
            showlegend=False
        ),
        row=3, col=1
    )

    # Layout with range slider only on bottom
    fig.update_layout(
        height=900,
        hovermode='x unified',
        template='plotly_white',
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
        xaxis3=dict(
            rangeslider=dict(visible=True, thickness=0.05),
            type='date'
        )
    )

    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="Market Cap ($)", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)

    fig.show()


def plot_target_distribution(df: pd.DataFrame, target_col: str = 'next_day_high', date_col: str = 'timeOpen', bins: int = 60):
    """Plot target variable: time series, distribution, and boxplot"""
    print("Target Variable Statistics:")
    print(df[target_col].describe())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Time series plot
    axes[0].plot(df[date_col], df[target_col], linewidth=1, alpha=0.8)
    axes[0].set_title('Next Day High Price Over Time')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Price ($)')
    axes[0].grid(True, alpha=0.3)

    # Histogram with shared y-axis
    sns.histplot(y=df[target_col], bins=bins, kde=True, ax=axes[1])
    axes[1].set_title('Target Distribution')
    axes[1].set_ylabel('Price ($)')
    axes[1].set_xlabel('Frequency')

    # Boxplot with shared y-axis
    sns.boxplot(y=df[target_col], ax=axes[2])
    axes[2].set_title('Target Boxplot')
    axes[2].set_ylabel('Price ($)')

    # Share y-axis between histogram and boxplot
    axes[1].set_ylim(axes[0].get_ylim())
    axes[2].set_ylim(axes[0].get_ylim())

    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, figsize: Tuple[int, int] = (10, 8)):
    """Plot correlation heatmap"""
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                square=True, linewidths=1)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()


def plot_stationarity_comparison(df: pd.DataFrame, date_col: str = 'timeOpen',
                                  price_col: str = 'close', return_col: str = 'close_return'):
    """Compare non-stationary price vs stationary returns"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Price (non-stationary)
    axes[0].plot(df[date_col], df[price_col], linewidth=1)
    axes[0].set_title('Close Price (Non-Stationary)')
    axes[0].set_ylabel('Price ($)')
    axes[0].grid(True, alpha=0.3)

    # Returns (more stationary)
    df_clean = df.dropna(subset=[return_col])
    axes[1].plot(df_clean[date_col], df_clean[return_col], linewidth=0.8, alpha=0.7)
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=1)
    axes[1].set_title('Daily Returns (More Stationary)')
    axes[1].set_ylabel('Return (%)')
    axes[1].set_xlabel('Date')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_feature_importance_comparison(correlation_scores: pd.Series,
                                       mi_scores: pd.Series,
                                       rf_scores: pd.Series = None,
                                       top_n: int = 15):
    """
    Compare feature importance from multiple methods

    Args:
        correlation_scores: Absolute correlation with target
        mi_scores: Mutual Information scores
        rf_scores: Random Forest feature importance (optional)
        top_n: Number of top features to display
    """
    n_methods = 2 if rf_scores is None else 3
    fig, axes = plt.subplots(1, n_methods, figsize=(6*n_methods, 8))

    if n_methods == 2:
        axes = [axes[0], axes[1]]

    # Plot 1: Correlation
    top_corr = correlation_scores.nlargest(top_n).sort_values()
    axes[0].barh(range(len(top_corr)), top_corr.values, color='steelblue')
    axes[0].set_yticks(range(len(top_corr)))
    axes[0].set_yticklabels(top_corr.index, fontsize=9)
    axes[0].set_xlabel('Absolute Correlation', fontsize=10)
    axes[0].set_title(f'Top {top_n} Features by Correlation', fontsize=11, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)

    # Plot 2: Mutual Information
    top_mi = mi_scores.nlargest(top_n).sort_values()
    axes[1].barh(range(len(top_mi)), top_mi.values, color='coral')
    axes[1].set_yticks(range(len(top_mi)))
    axes[1].set_yticklabels(top_mi.index, fontsize=9)
    axes[1].set_xlabel('Mutual Information Score', fontsize=10)
    axes[1].set_title(f'Top {top_n} Features by Mutual Information', fontsize=11, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)

    # Plot 3: Random Forest (optional)
    if rf_scores is not None:
        top_rf = rf_scores.nlargest(top_n).sort_values()
        axes[2].barh(range(len(top_rf)), top_rf.values, color='mediumseagreen')
        axes[2].set_yticks(range(len(top_rf)))
        axes[2].set_yticklabels(top_rf.index, fontsize=9)
        axes[2].set_xlabel('Feature Importance', fontsize=10)
        axes[2].set_title(f'Top {top_n} Features by Random Forest', fontsize=11, fontweight='bold')
        axes[2].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_technical_indicators(df: pd.DataFrame, date_col: str = 'timeOpen', n_days: int = 365):
    """Plot technical indicators with price"""
    df_plot = df.tail(n_days) if n_days else df.copy()

    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    # Plot 1: Price with Moving Averages
    axes[0].plot(df_plot[date_col], df_plot['close'], label='Close', linewidth=1.5, color='black')
    axes[0].plot(df_plot[date_col], df_plot['ma_7'], label='MA(7)', linewidth=1, alpha=0.7)
    axes[0].plot(df_plot[date_col], df_plot['ma_30'], label='MA(30)', linewidth=1, alpha=0.7)
    axes[0].plot(df_plot[date_col], df_plot['ma_90'], label='MA(90)', linewidth=1, alpha=0.7)
    axes[0].set_ylabel('Price ($)')
    axes[0].set_title('Price with Moving Averages', fontweight='bold')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Bollinger Bands
    axes[1].plot(df_plot[date_col], df_plot['close'], label='Close', linewidth=1.5, color='black')
    axes[1].fill_between(df_plot[date_col], df_plot['bb_upper'], df_plot['bb_lower'],
                         alpha=0.2, color='gray', label='Bollinger Bands')
    axes[1].plot(df_plot[date_col], df_plot['bb_middle'], label='BB Middle',
                linewidth=1, linestyle='--', color='blue')
    axes[1].set_ylabel('Price ($)')
    axes[1].set_title('Bollinger Bands', fontweight='bold')
    axes[1].legend(loc='upper left')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: RSI
    axes[2].plot(df_plot[date_col], df_plot['rsi'], linewidth=1, color='purple')
    axes[2].axhline(y=70, color='r', linestyle='--', linewidth=1, label='Overbought (70)')
    axes[2].axhline(y=30, color='g', linestyle='--', linewidth=1, label='Oversold (30)')
    axes[2].fill_between(df_plot[date_col], 30, 70, alpha=0.1, color='gray')
    axes[2].set_ylabel('RSI')
    axes[2].set_ylim([0, 100])
    axes[2].set_title('Relative Strength Index (RSI)', fontweight='bold')
    axes[2].legend(loc='upper left')
    axes[2].grid(True, alpha=0.3)

    # Plot 4: MACD
    axes[3].plot(df_plot[date_col], df_plot['macd'], label='MACD', linewidth=1, color='blue')
    axes[3].plot(df_plot[date_col], df_plot['macd_signal'], label='Signal', linewidth=1, color='red')
    axes[3].bar(df_plot[date_col], df_plot['macd_diff'], label='Histogram',
               alpha=0.3, color='gray', width=1)
    axes[3].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[3].set_ylabel('MACD')
    axes[3].set_xlabel('Date')
    axes[3].set_title('MACD (Moving Average Convergence Divergence)', fontweight='bold')
    axes[3].legend(loc='upper left')
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_features_correlation(correlations: pd.Series, target_col: str = 'next_day_high'):
    """Plot correlation bar chart for all features"""
    # Sort by absolute value (descending) - largest at top
    correlations_sorted = correlations.reindex(correlations.abs().sort_values(ascending=True).index)

    # Adjust figure size based on number of features
    n_features = len(correlations_sorted)
    fig_height = max(12, n_features * 0.25)  # At least 12, scale with features

    plt.figure(figsize=(12, fig_height))
    correlations_sorted.plot(kind='barh', color='skyblue', edgecolor='black')
    plt.title(f'Feature Correlation with {target_col} (n={n_features})',
              fontweight='bold', fontsize=14)
    plt.xlabel('Correlation Coefficient', fontsize=12)
    plt.ylabel('Features', fontsize=10)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.grid(True, alpha=0.3, axis='x')

    # Adjust y-axis tick label font size based on number of features
    if n_features > 30:
        plt.yticks(fontsize=7)
    elif n_features > 15:
        plt.yticks(fontsize=8)
    else:
        plt.yticks(fontsize=9)

    plt.tight_layout()
    plt.show()


def plot_raw_features_distribution(df: pd.DataFrame):
    """Plot distribution of raw features"""
    raw_features = ['open', 'high', 'low', 'close', 'volume', 'marketCap']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    for idx, col in enumerate(raw_features):
        ax = axes[idx]
        df[col].hist(bins=50, ax=ax, edgecolor='black', alpha=0.7, color='steelblue')
        ax.set_title(f'{col} Distribution', fontweight='bold')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_regression_results(y_true, y_pred, figsize=(15, 10)):
    """
    Unified visualization for any regression model

    Args:
        y_true: True values (original scale)
        y_pred: Predicted values (original scale)
        figsize: Figure size
    """
    from sklearn.metrics import mean_absolute_error, r2_score

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Plot 1: Predictions vs Actual
    axes[0, 0].scatter(y_true, y_pred, alpha=0.5, s=20)
    axes[0, 0].plot([y_true.min(), y_true.max()],
                   [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Next-Day High Price ($)', fontsize=12)
    axes[0, 0].set_ylabel('Predicted Next-Day High Price ($)', fontsize=12)
    r2 = r2_score(y_true, y_pred)
    axes[0, 0].set_title(f'Test Set: Predictions vs Actual (R²={r2:.4f})', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Residuals
    residuals = y_true - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.5, s=20)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted Price ($)', fontsize=12)
    axes[0, 1].set_ylabel('Residuals ($)', fontsize=12)
    axes[0, 1].set_title('Residual Plot', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Time series
    axes[1, 0].plot(range(len(y_true)), y_true, label='Actual', linewidth=2, alpha=0.7)
    axes[1, 0].plot(range(len(y_pred)), y_pred, label='Predicted', linewidth=2, alpha=0.7)
    axes[1, 0].set_xlabel('Test Sample Index', fontsize=12)
    axes[1, 0].set_ylabel('Next-Day High Price ($)', fontsize=12)
    axes[1, 0].set_title('Time Series: Actual vs Predicted', fontsize=14)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Error distribution
    mae = mean_absolute_error(y_true, y_pred)
    axes[1, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1, 1].set_xlabel('Prediction Error ($)', fontsize=12)
    axes[1, 1].set_ylabel('Frequency', fontsize=12)
    axes[1, 1].set_title(f'Error Distribution (MAE=${mae:,.2f})', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def analyze_optimal_feature_count(df, target_col='next_day_high',
                                   N_values=None,
                                   test_ratio=0.2):
    """
    Systematically test different top-N feature counts using Linear Regression

    Args:
        df: DataFrame with all features and target
        target_col: Target column name
        N_values: List of N values to test (default: [3, 5, 7, 10, 12, 15, 20, 25, 30, 40, 50, 60])
        test_ratio: Test set ratio for chronological split

    Returns:
        pd.DataFrame: Results with columns ['N', 'R2', 'MAE', 'MAPE']
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    if N_values is None:
        N_values = [3, 5, 7, 10, 12, 15, 20, 25, 30, 40, 50, 60]

    # Get all features except date and target
    all_features = [col for col in df.columns if col not in ['date', target_col]]

    # Calculate correlations with target
    correlations = df[all_features + [target_col]].corr()[target_col].drop(target_col)
    correlations_sorted = correlations.abs().sort_values(ascending=False)

    # Chronological split
    n_test = int(len(df) * test_ratio)
    train_df = df.iloc[:-n_test].copy()
    test_df = df.iloc[-n_test:].copy()

    y_train = train_df[target_col].values
    y_test = test_df[target_col].values

    results = []

    print("Testing different feature counts...")
    print(f"{'N':>4} {'R2':>10} {'MAE':>12} {'MAPE':>8}")
    print("-" * 40)

    for N in N_values:
        # Select top N features
        top_features = correlations_sorted.head(N).index.tolist()

        # Prepare data
        X_train = train_df[top_features].values
        X_test = test_df[top_features].values

        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        results.append({'N': N, 'R2': r2, 'MAE': mae, 'MAPE': mape})
        print(f"{N:4d} {r2:10.6f} ${mae:10,.2f} {mape:7.2f}%")

    print("-" * 40)

    results_df = pd.DataFrame(results)

    # Print optimal results
    best_r2_idx = results_df['R2'].idxmax()
    best_mape_idx = results_df['MAPE'].idxmin()
    best_mae_idx = results_df['MAE'].idxmin()

    print(f"\nOptimal Feature Count:")
    print(f"  Best R2:   N={results_df.loc[best_r2_idx, 'N']:2.0f}  (R2={results_df.loc[best_r2_idx, 'R2']:.6f})")
    print(f"  Best MAPE: N={results_df.loc[best_mape_idx, 'N']:2.0f}  (MAPE={results_df.loc[best_mape_idx, 'MAPE']:.2f}%)")
    print(f"  Best MAE:  N={results_df.loc[best_mae_idx, 'N']:2.0f}  (MAE=${results_df.loc[best_mae_idx, 'MAE']:,.2f})")

    return results_df


def plot_optimal_feature_count(results_df, figsize=(18, 5)):
    """
    Plot performance curves for different feature counts

    Args:
        results_df: DataFrame from analyze_optimal_feature_count()
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # R2 curve
    axes[0].plot(results_df['N'], results_df['R2'], marker='o', linewidth=2,
                markersize=8, color='#2E86AB')
    best_r2_idx = results_df['R2'].idxmax()
    axes[0].axvline(results_df.loc[best_r2_idx, 'N'], color='red', linestyle='--',
                   alpha=0.6, label=f"Optimal N={int(results_df.loc[best_r2_idx, 'N'])}")
    axes[0].set_xlabel('Number of Top Features (N)', fontsize=12)
    axes[0].set_ylabel('R2 Score', fontsize=12)
    axes[0].set_title('R2 vs Feature Count', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # MAPE curve
    axes[1].plot(results_df['N'], results_df['MAPE'], marker='o', linewidth=2,
                markersize=8, color='#A23B72')
    best_mape_idx = results_df['MAPE'].idxmin()
    axes[1].axvline(results_df.loc[best_mape_idx, 'N'], color='red', linestyle='--',
                   alpha=0.6, label=f"Optimal N={int(results_df.loc[best_mape_idx, 'N'])}")
    axes[1].set_xlabel('Number of Top Features (N)', fontsize=12)
    axes[1].set_ylabel('MAPE (%)', fontsize=12)
    axes[1].set_title('MAPE vs Feature Count', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # MAE curve
    axes[2].plot(results_df['N'], results_df['MAE'], marker='o', linewidth=2,
                markersize=8, color='#F18F01')
    best_mae_idx = results_df['MAE'].idxmin()
    axes[2].axvline(results_df.loc[best_mae_idx, 'N'], color='red', linestyle='--',
                   alpha=0.6, label=f"Optimal N={int(results_df.loc[best_mae_idx, 'N'])}")
    axes[2].set_xlabel('Number of Top Features (N)', fontsize=12)
    axes[2].set_ylabel('MAE ($)', fontsize=12)
    axes[2].set_title('MAE vs Feature Count', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.show()
