"""
Bitcoin data processing and feature engineering module
"""

import pandas as pd
import numpy as np
import glob
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

try:
    import pandas_ta_classic as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    warnings.warn("pandas-ta-classic not installed. Technical indicators will not be available.")


# ============================================================================
# OPTIMAL FEATURE SELECTION (Systematic Search Results)
# ============================================================================
# Based on systematic search across N=3-60, optimal N=30 features achieve:
# Best balance between model performance and complexity
# Features selected by absolute correlation with next_day_high target

OPTIMAL_FEATURES_30 = [
    'close',          # 0.9995 - Current day closing price
    'VWAP',           # 0.9994 - Volume Weighted Average Price
    'high',           # 0.9993 - Current day high
    'low',            # 0.9989 - Current day low
    'marketCap',      # 0.9989 - Market capitalization
    'open',           # 0.9987 - Current day opening price
    'TEMA_20',        # 0.9984 - Triple Exponential Moving Average (20-period)
    'SMA_7',          # 0.9979 - Simple Moving Average (7-period)
    'DEMA_20',        # 0.9978 - Double Exponential Moving Average (20-period)
    'HMA_20',         # 0.9978 - Hull Moving Average (20-period)
    'EMA_12',         # 0.9972 - Exponential Moving Average (12-period)
    'KCBe_20_2.0',    # 0.9951 - Keltner Channel Basis (20-period, 2.0 scalar)
    'DCM_20_20',      # 0.9944 - Donchian Channel Middle (20-period)
    'KCUe_20_2.0',    # 0.9942 - Keltner Channel Upper (20-period, 2.0 scalar)
    'KCLe_20_2.0',    # 0.9942 - Keltner Channel Lower (20-period, 2.0 scalar)
    'DCU_20_20',      # 0.9940 - Donchian Channel Upper (20-period)
    'BBM_20_2.0',     # 0.9935 - Bollinger Band Middle (20-period, 2.0 std)
    'SMA_20',         # 0.9935 - Simple Moving Average (20-period)
    'EMA_26',         # 0.9935 - Exponential Moving Average (26-period)
    'BBU_20_2.0',     # 0.9934 - Bollinger Band Upper (20-period, 2.0 std)
    'DCL_20_20',      # 0.9894 - Donchian Channel Lower (20-period)
    'BBL_20_2.0',     # 0.9865 - Bollinger Band Lower (20-period, 2.0 std)
    'EMA_50',         # 0.9865 - Exponential Moving Average (50-period)
    'SMA_50',         # 0.9817 - Simple Moving Average (50-period)
    'SMA_200',        # 0.9262 - Simple Moving Average (200-period)
    'ATRr_14',        # 0.9019 - Average True Range Ratio (14-period)
    'AD',             # 0.8741 - Accumulation/Distribution Line
    'OBV',            # 0.8286 - On-Balance Volume
    'volume',         # 0.6493 - Trading volume
    'MACDs_12_26_9',  # 0.3725 - MACD Signal Line (12, 26, 9-period)
]


class DataLoader:
    """Data loading and combining"""

    def __init__(self, raw_data_dir: str = "data/raw"):
        project_root = Path(__file__).parent.parent
        self.raw_data_dir = project_root / raw_data_dir
        self.interim_dir = project_root / "data" / "interim"
        self.interim_dir.mkdir(parents=True, exist_ok=True)

    def load_raw_data(self, pattern: str = "Bitcoin_*.csv", save_to_interim: bool = True) -> pd.DataFrame:
        """
        Load and combine all Bitcoin CSV files from raw data directory

        Parameters:
        -----------
        pattern : str
            File pattern to match (default: "Bitcoin_*.csv")
        save_to_interim : bool
            Save combined data to data/interim/bitcoin_combined.csv

        Returns:
        --------
        pd.DataFrame
            Combined Bitcoin dataset with parsed datetime
        """
        csv_files = sorted(glob.glob(str(self.raw_data_dir / pattern)))

        if not csv_files:
            raise FileNotFoundError(f"No files found matching {pattern} in {self.raw_data_dir}")

        all_data = []
        for file in csv_files:
            df = pd.read_csv(file, sep=';', encoding='utf-8-sig')
            all_data.append(df)

        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df['timeOpen'] = pd.to_datetime(combined_df['timeOpen'])
        combined_df = combined_df.sort_values('timeOpen').reset_index(drop=True)

        print(f"Loaded {len(csv_files)} files: {len(combined_df)} records from {combined_df['timeOpen'].min().date()} to {combined_df['timeOpen'].max().date()}")

        if save_to_interim:
            output_path = self.interim_dir / 'bitcoin_combined.csv'
            combined_df.to_csv(output_path, index=False)

        return combined_df

    def load_combined_data(self) -> pd.DataFrame:
        """Load pre-combined Bitcoin data from interim directory"""
        file_path = self.interim_dir / 'bitcoin_combined.csv'

        if not file_path.exists():
            raise FileNotFoundError(f"Combined data not found: {file_path}\nRun load_raw_data() first.")

        df = pd.read_csv(file_path)
        df['timeOpen'] = pd.to_datetime(df['timeOpen'])

        print(f"Loaded combined data: {len(df)} records")
        return df


class FeatureEngineer:
    """Feature engineering and data transformation"""

    def __init__(self):
        project_root = Path(__file__).parent.parent
        self.interim_dir = project_root / "data" / "interim"
        self.interim_dir.mkdir(parents=True, exist_ok=True)

    def create_target_variable(self, df: pd.DataFrame, target_col: str = 'high',
                              save_to_interim: bool = True) -> pd.DataFrame:
        """
        Create next day's HIGH price as target variable

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with OHLC data
        target_col : str
            Column to shift for creating target (default: 'high')
        save_to_interim : bool
            Save result to interim directory

        Returns:
        --------
        pd.DataFrame
            DataFrame with 'next_day_high' column added
        """
        df = df.copy()
        df['next_day_high'] = df[target_col].shift(-1)
        df = df.dropna(subset=['next_day_high']).copy()

        print(f"Created target 'next_day_high' ({len(df)} samples)")

        if save_to_interim:
            interim_file = self.interim_dir / "bitcoin_with_target.csv"
            df.to_csv(interim_file, index=False)

        return df

    def clean_timestamp_columns(self, df: pd.DataFrame, save_to_interim: bool = True) -> pd.DataFrame:
        """
        Clean timestamp columns and keep only essential date column

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with timestamp columns
        save_to_interim : bool
            Save cleaned data to interim directory

        Returns:
        --------
        pd.DataFrame
            Cleaned dataframe with only 'date' column for time reference
        """
        df_clean = df.copy()

        # Rename timeOpen to date
        if 'timeOpen' in df_clean.columns:
            df_clean = df_clean.rename(columns={'timeOpen': 'date'})

        # Remove redundant timestamp columns
        cols_to_drop = ['timeClose', 'timeHigh', 'timeLow', 'timestamp', 'name']
        cols_to_drop = [col for col in cols_to_drop if col in df_clean.columns]

        if cols_to_drop:
            df_clean = df_clean.drop(columns=cols_to_drop)

        print(f"Cleaned data: {len(df_clean)} samples, {len(df_clean.columns)} columns")

        if save_to_interim:
            interim_file = self.interim_dir / "bitcoin_cleaned.csv"
            df_clean.to_csv(interim_file, index=False)

        return df_clean

    def add_technical_indicators(self, df: pd.DataFrame,
                                 handle_missing: bool = True,
                                 save_to_interim: bool = True) -> pd.DataFrame:
        """
        Add comprehensive technical indicators using pandas-ta-classic

        Categories:
        - Trend: SMA, EMA, DEMA, TEMA, HMA, VWAP
        - Momentum: RSI, MACD, Stochastic, Williams %R, ROC, CCI, MFI
        - Volatility: Bollinger Bands, ATR, Keltner Channels, Donchian
        - Volume: OBV, AD, ADOSC, CMF
        - Trend Strength: ADX, Aroon
        - Custom: Price ratios, BB position, volume surge, momentum

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with OHLC data
        handle_missing : bool
            Remove warmup rows with NaN values (default: True)
        save_to_interim : bool
            Save to interim directory (default: True)

        Returns:
        --------
        pd.DataFrame
            DataFrame with 54 technical indicators added (SUPERTREND removed)
        """
        if not PANDAS_TA_AVAILABLE:
            raise ImportError("pandas-ta-classic is not installed. Please run: pip install pandas-ta-classic")

        df = df.copy()
        initial_cols = len(df.columns)

        # === 1. TREND INDICATORS (Overlap Studies) ===
        df.ta.sma(length=7, append=True)
        df.ta.sma(length=20, append=True)
        df.ta.sma(length=50, append=True)
        df.ta.sma(length=200, append=True)
        df.ta.ema(length=12, append=True)
        df.ta.ema(length=26, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.dema(length=20, append=True)
        df.ta.tema(length=20, append=True)
        df.ta.hma(length=20, append=True)

        # VWAP
        try:
            if 'date' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
                df_temp = df.set_index(pd.to_datetime(df['date']))
                vwap_values = df_temp.ta.vwap()
                df['VWAP'] = vwap_values.values
            else:
                df.ta.vwap(append=True)
        except:
            pass  # Skip VWAP if error

        # === 2. MOMENTUM INDICATORS ===
        df.ta.rsi(length=14, append=True)
        df.ta.rsi(length=21, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.stoch(k=14, d=3, smooth_k=3, append=True)
        df.ta.willr(length=14, append=True)
        df.ta.roc(length=10, append=True)
        df.ta.roc(length=21, append=True)
        df.ta.cci(length=14, append=True)
        df.ta.mfi(length=14, append=True)

        # === 3. VOLATILITY INDICATORS ===
        df.ta.bbands(length=20, std=2, append=True)
        df.ta.atr(length=14, append=True)
        df.ta.natr(length=14, append=True)
        df.ta.kc(length=20, scalar=2, append=True)
        df.ta.donchian(lower_length=20, upper_length=20, append=True)

        # === 4. VOLUME INDICATORS ===
        df.ta.obv(append=True)
        df.ta.ad(append=True)
        df.ta.adosc(fast=3, slow=10, append=True)
        df.ta.cmf(length=20, append=True)

        # === 5. TREND STRENGTH INDICATORS ===
        df.ta.adx(length=14, append=True)
        df.ta.aroon(length=25, append=True)
        # SUPERTREND removed - causes NaN values when trend direction changes

        # === 6. CUSTOM DERIVED FEATURES ===
        if 'SMA_20' in df.columns:
            df['price_to_sma20'] = (df['close'] - df['SMA_20']) / df['SMA_20'] * 100
            df['price_to_sma50'] = (df['close'] - df['SMA_50']) / df['SMA_50'] * 100
        if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
            df['sma20_50_ratio'] = df['SMA_20'] / df['SMA_50']
        if 'BBL_20_2.0' in df.columns and 'BBU_20_2.0' in df.columns:
            df['bb_width_pct'] = (df['BBU_20_2.0'] - df['BBL_20_2.0']) / df['BBM_20_2.0'] * 100
            df['bb_position'] = (df['close'] - df['BBL_20_2.0']) / (df['BBU_20_2.0'] - df['BBL_20_2.0'])
        if 'SMA_20' in df.columns:
            volume_sma = df['volume'].rolling(window=20).mean()
            df['volume_surge'] = df['volume'] / volume_sma
        df['price_momentum_5'] = df['close'].pct_change(5) * 100
        df['price_momentum_10'] = df['close'].pct_change(10) * 100

        final_cols = len(df.columns)
        added_features = final_cols - initial_cols

        print(f"Added {added_features} technical indicators")

        # Save interim dataset with indicators (may have NaN)
        if save_to_interim:
            interim_file = self.interim_dir / "bitcoin_with_technical_indicators.csv"
            df.to_csv(interim_file, index=False)

        # Handle missing values if requested
        if handle_missing:
            df = self.handle_missing_values(df, strategy='comprehensive',
                                           warmup_rows=199,
                                           save_to_interim=True)

        return df

    def handle_missing_values(self, df: pd.DataFrame,
                             strategy: str = 'comprehensive',
                             warmup_rows: int = 199,
                             save_to_interim: bool = True) -> pd.DataFrame:
        """
        Handle missing values from technical indicator calculations

        Strategy: Remove initial warmup rows (e.g., 199 days for SMA_200)

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with potential NaN values from indicators
        strategy : str
            'comprehensive': Remove warmup_rows from beginning
        warmup_rows : int
            Number of initial rows to remove (default: 199 for SMA_200)
        save_to_interim : bool
            Save cleaned data to interim directory

        Returns:
        --------
        pd.DataFrame
            DataFrame with missing values handled
        """
        df_clean = df.copy()

        if strategy == 'comprehensive':
            # Remove warmup rows from beginning
            df_clean = df_clean.iloc[warmup_rows:].reset_index(drop=True)
            print(f"Removed {warmup_rows} warmup rows ({len(df_clean)} samples remaining)")

        if save_to_interim:
            interim_file = self.interim_dir / "bitcoin_clean.csv"
            df_clean.to_csv(interim_file, index=False)

        return df_clean


class DataAnalyzer:
    """Statistical and correlation analysis"""

    def analyze_features_correlation(self, df: pd.DataFrame, target_col: str = 'next_day_high') -> pd.Series:
        """
        Analyze correlation between all features and target

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with features and target
        target_col : str
            Target column name

        Returns:
        --------
        pd.Series
            Correlations sorted by absolute value (descending)
        """
        # Get all features except date and target
        exclude_cols = ['date', target_col]
        available_features = [col for col in df.columns if col not in exclude_cols]

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")

        # Calculate correlations
        correlations = df[available_features + [target_col]].corr()[target_col].drop(target_col)

        # Sort by absolute value
        correlations = correlations.reindex(correlations.abs().sort_values(ascending=False).index)

        print(f"\nFeature Correlation with '{target_col}' (Total: {len(available_features)} features):")
        print("="*60)
        for feature, corr in correlations.items():
            print(f"  {feature:15s}: {corr:7.4f}")
        print("="*60)

        return correlations

    def analyze_raw_features_statistics(self, df: pd.DataFrame) -> dict:
        """
        Analyze statistical properties of raw features

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with raw features

        Returns:
        --------
        dict
            Dictionary with outlier statistics for each feature
        """
        raw_features = ['open', 'high', 'low', 'close', 'volume', 'marketCap']
        available_features = [f for f in raw_features if f in df.columns]

        outlier_stats = {}

        print(f"\nRaw Features Statistical Analysis:")
        print("="*60)

        for feature in available_features:
            data = df[feature].values

            # Calculate IQR-based outliers
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = ((data < lower_bound) | (data > upper_bound)).sum()
            outlier_pct = (outliers / len(data)) * 100

            outlier_stats[feature] = {
                'count': len(data),
                'outliers': outliers,
                'outlier_pct': outlier_pct,
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR
            }

            print(f"  {feature:15s}: {outliers:4d} outliers ({outlier_pct:5.2f}%)")

        print("="*60)

        return outlier_stats


class DataTransformer:
    """Data transformation and normalization"""

    def __init__(self):
        project_root = Path(__file__).parent.parent
        self.processed_dir = project_root / "data" / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def time_series_split(self, df: pd.DataFrame,
                         train_ratio: float = 0.6,
                         val_ratio: float = 0.2,
                         test_ratio: float = 0.2) -> tuple:
        """Split time series data chronologically into train/val/test sets"""
        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

        n = len(df)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()

        date_col = 'date' if 'date' in df.columns else 'timeOpen'

        def format_date(dt):
            return pd.to_datetime(dt).strftime('%Y-%m-%d')

        train_start = format_date(train_df[date_col].min())
        train_end = format_date(train_df[date_col].max())
        val_start = format_date(val_df[date_col].min())
        val_end = format_date(val_df[date_col].max())
        test_start = format_date(test_df[date_col].min())
        test_end = format_date(test_df[date_col].max())

        print(f"\nSplit: Train {len(train_df)} | Val {len(val_df)} | Test {len(test_df)}")
        print(f"  Train: {train_start} to {train_end}")
        print(f"  Val:   {val_start} to {val_end}")
        print(f"  Test:  {test_start} to {test_end}")

        return train_df, val_df, test_df

    def log_transform_and_scale(self, train_df: pd.DataFrame,
                                val_df: pd.DataFrame,
                                test_df: pd.DataFrame,
                                target_col: str = 'next_day_high',
                                save_to_processed: bool = False) -> tuple:
        """
        Log transformation + MinMaxScaler normalization - UNIFIED FORMAT

        *** NATIVE SUPPORT FOR UNIFIED DATA - NO X/Y SEPARATION ***

        Following Time-Series-Library official format:
        - Returns complete DataFrames with ALL features (including target)
        - Target column placed FIRST for easy indexing in Dataset class
        - data_x and data_y will be the SAME in TimesNet Dataset

        *** USES OPTIMAL 30 FEATURES ONLY ***

        Parameters:
        -----------
        train_df, val_df, test_df : pd.DataFrame
            Train, validation and test datasets (with ALL columns)
        target_col : str
            Name of target column (default: 'next_day_high')
        save_to_processed : bool
            Save processed datasets to data/processed/

        Returns:
        --------
        tuple
            (train_data, val_data, test_data, scaler)
            Each DataFrame has shape (n_samples, n_features+1) with target first
        """
        # Copy dataframes
        train = train_df.copy()
        val = val_df.copy()
        test = test_df.copy()

        # Keep only optimal features + target + date (if exists)
        keep_cols = OPTIMAL_FEATURES_30 + [target_col]
        if 'date' in train.columns:
            keep_cols = ['date'] + keep_cols

        train = train[keep_cols].copy()
        val = val[keep_cols].copy()
        test = test[keep_cols].copy()

        print(f"\nUsing {len(OPTIMAL_FEATURES_30)} optimal features for modeling")

        # ======================================================================
        # STEP 1: Log Transformation
        # ======================================================================

        # Identify features to apply log transformation
        # 1. Raw price and volume features
        raw_price_volume = ['open', 'high', 'low', 'close', 'volume', 'marketCap']

        # 2. Price-based technical indicators (moving averages, bands, channels)
        price_based_indicators = [
            'SMA_7', 'SMA_20', 'SMA_50', 'SMA_200',
            'EMA_12', 'EMA_26', 'EMA_50',
            'DEMA_20', 'TEMA_20', 'HMA_20', 'VWAP',
            'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0',
            'KCLe_20_2.0', 'KCBe_20_2.0', 'KCUe_20_2.0',
            'DCL_20_20', 'DCM_20_20', 'DCU_20_20'
            # SUPERTREND removed - not generated anymore
        ]

        # 3. Cumulative/volume-based indicators (can have extreme values)
        cumulative_indicators = ['OBV', 'AD', 'ADOSC_3_10']

        # 4. Volatility indicators with large ranges
        volatility_indicators = ['ATRr_14']

        # 5. Momentum indicators with large ranges (MACD series)
        momentum_indicators = ['MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9']

        # Combine all features that need log transformation
        log_features = (raw_price_volume + price_based_indicators +
                       cumulative_indicators + volatility_indicators + momentum_indicators)

        # Apply log transformation
        transformed_count = 0
        signed_log_count = 0

        for col in log_features:
            if col in train.columns:
                # Check if feature has positive values only
                if train[col].min() > 0 and val[col].min() > 0 and test[col].min() > 0:
                    train[col] = np.log1p(train[col])
                    val[col] = np.log1p(val[col])
                    test[col] = np.log1p(test[col])
                    transformed_count += 1
                elif train[col].min() < 0:
                    # For indicators with negative values, use signed log
                    for df in [train, val, test]:
                        df[col] = np.sign(df[col]) * np.log1p(np.abs(df[col]))
                    transformed_count += 1
                    signed_log_count += 1

        print(f"\nLog-transformed {transformed_count} features ({signed_log_count} signed)")

        # Apply log transformation to target variable
        train[target_col] = np.log1p(train[target_col])
        val[target_col] = np.log1p(val[target_col])
        test[target_col] = np.log1p(test[target_col])

        # ======================================================================
        # STEP 2: Prepare Unified Data Format
        # ======================================================================

        # Save date columns separately (for visualization, not for model input)
        train_dates = train['date'].copy() if 'date' in train.columns else None
        val_dates = val['date'].copy() if 'date' in val.columns else None
        test_dates = test['date'].copy() if 'date' in test.columns else None

        # Select all columns except 'date'
        exclude_cols = ['date']
        data_cols = [col for col in train.columns if col not in exclude_cols]

        # Reorder: target column FIRST, then all other features
        # This makes indexing easier: data[:, 0] = target, data[:, 1:] = features
        if target_col in data_cols:
            data_cols.remove(target_col)
        data_cols = [target_col] + data_cols

        # Extract unified data arrays
        train_data = train[data_cols].values
        val_data = val[data_cols].values
        test_data = test[data_cols].values

        # ======================================================================
        # STEP 3: MinMaxScaler Normalization (UNIFIED - No X/y separation!)
        # ======================================================================

        scaler = MinMaxScaler()

        # Fit scaler on training set only (avoid data leakage)
        scaler.fit(train_data)

        # Transform all datasets using training set statistics
        train_data_scaled = scaler.transform(train_data)
        val_data_scaled = scaler.transform(val_data)
        test_data_scaled = scaler.transform(test_data)

        print(f"Normalized {train_data_scaled.shape[1]} features to [0, 1]")

        # Convert back to DataFrames
        train_data_df = pd.DataFrame(train_data_scaled, columns=data_cols)
        val_data_df = pd.DataFrame(val_data_scaled, columns=data_cols)
        test_data_df = pd.DataFrame(test_data_scaled, columns=data_cols)

        # Add date columns back (for visualization, stored separately)
        # Convert to date-only format (remove timestamp)
        if train_dates is not None:
            train_data_df.insert(0, 'date', pd.to_datetime(train_dates.values).date)
        if val_dates is not None:
            val_data_df.insert(0, 'date', pd.to_datetime(val_dates.values).date)
        if test_dates is not None:
            test_data_df.insert(0, 'date', pd.to_datetime(test_dates.values).date)

        # ======================================================================
        # STEP 4: Optionally Save
        # ======================================================================

        if save_to_processed:
            train_data_df.to_csv(self.processed_dir / 'train.csv', index=False)
            val_data_df.to_csv(self.processed_dir / 'val.csv', index=False)
            test_data_df.to_csv(self.processed_dir / 'test.csv', index=False)

            # Save scaler
            import joblib
            joblib.dump(scaler, self.processed_dir / 'scaler.pkl')

            # Save metadata
            metadata = {
                'target_col': target_col,
                'n_features': len(data_cols) - 1,
                'feature_cols': data_cols[1:],  # Exclude target
                'all_cols': data_cols,
            }
            joblib.dump(metadata, self.processed_dir / 'metadata.pkl')

        return (train_data_df, val_data_df, test_data_df, scaler)


class BitcoinDataProcessor:
    """Unified Bitcoin data processing pipeline"""

    def __init__(self, raw_data_dir: str = "data/raw"):
        # Initialize all sub-components
        self.loader = DataLoader(raw_data_dir)
        self.engineer = FeatureEngineer()
        self.analyzer = DataAnalyzer()
        self.transformer = DataTransformer()

        # Expose directories for compatibility
        self.raw_data_dir = self.loader.raw_data_dir
        self.interim_dir = self.loader.interim_dir
        self.processed_dir = self.transformer.processed_dir

    # Data Loading (delegate to DataLoader)
    def load_raw_data(self, pattern: str = "Bitcoin_*.csv", save_to_interim: bool = True) -> pd.DataFrame:
        return self.loader.load_raw_data(pattern, save_to_interim)

    def load_combined_data(self) -> pd.DataFrame:
        return self.loader.load_combined_data()

    # Feature Engineering (delegate to FeatureEngineer)
    def create_target_variable(self, df: pd.DataFrame, target_col: str = 'high',
                              save_to_interim: bool = True) -> pd.DataFrame:
        return self.engineer.create_target_variable(df, target_col, save_to_interim)

    def clean_timestamp_columns(self, df: pd.DataFrame, save_to_interim: bool = True) -> pd.DataFrame:
        return self.engineer.clean_timestamp_columns(df, save_to_interim)

    def add_technical_indicators(self, df: pd.DataFrame,
                                 handle_missing: bool = True,
                                 save_to_interim: bool = True) -> pd.DataFrame:
        return self.engineer.add_technical_indicators(df, handle_missing, save_to_interim)

    def handle_missing_values(self, df: pd.DataFrame,
                             strategy: str = 'comprehensive',
                             warmup_rows: int = 199,
                             save_to_interim: bool = True) -> pd.DataFrame:
        return self.engineer.handle_missing_values(df, strategy, warmup_rows, save_to_interim)

    # Data Analysis (delegate to DataAnalyzer)
    def analyze_features_correlation(self, df: pd.DataFrame, target_col: str = 'next_day_high') -> pd.Series:
        return self.analyzer.analyze_features_correlation(df, target_col)

    def analyze_raw_features_statistics(self, df: pd.DataFrame) -> dict:
        return self.analyzer.analyze_raw_features_statistics(df)

    # Data Transformation (delegate to DataTransformer) - UNIFIED FORMAT
    def time_series_split(self, df: pd.DataFrame,
                         train_ratio: float = 0.6,
                         val_ratio: float = 0.2,
                         test_ratio: float = 0.2) -> tuple:
        return self.transformer.time_series_split(df, train_ratio, val_ratio, test_ratio)

    def log_transform_and_scale(self, train_df: pd.DataFrame,
                                val_df: pd.DataFrame,
                                test_df: pd.DataFrame,
                                target_col: str = 'next_day_high',
                                save_to_processed: bool = False) -> tuple:
        """
        NATIVE UNIFIED FORMAT - NO X/Y SEPARATION

        Returns (train_data, val_data, test_data, scaler)
        Each DataFrame contains ALL features including target
        """
        return self.transformer.log_transform_and_scale(train_df, val_df, test_df,
                                                       target_col, save_to_processed)


# ============================================================================
# Data Preparation Functions
# ============================================================================

def prepare_linear_regression_data(train_df, val_df=None, test_df=None, target_col='next_day_high'):
    """
    Prepare data for linear regression (no standardization needed)

    *** USES OPTIMAL 30 FEATURES ONLY ***

    Args:
        train_df: Training DataFrame
        val_df: Optional validation DataFrame
        test_df: Optional test DataFrame
        target_col: Target column name

    Returns:
        dict: Dictionary with X_train, y_train, X_val, y_val, X_test, y_test, feature_cols
    """
    # Use predefined optimal 30 features
    feature_cols = OPTIMAL_FEATURES_30.copy()
    print(f"Using {len(feature_cols)} optimal features for linear regression")

    # Prepare training data
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values

    result = {
        'X_train': X_train,
        'y_train': y_train,
        'feature_cols': feature_cols
    }

    # Prepare validation data if provided
    if val_df is not None:
        X_val = val_df[feature_cols].values
        y_val = val_df[target_col].values
        result['X_val'] = X_val
        result['y_val'] = y_val

    # Prepare test data if provided
    if test_df is not None:
        X_test = test_df[feature_cols].values
        y_test = test_df[target_col].values
        result['X_test'] = X_test
        result['y_test'] = y_test

    return result


def inverse_transform_predictions(y_pred_scaled, scaler, n_features=12):
    """
    Inverse transform predictions from scaled space to original prices

    Args:
        y_pred_scaled: Predictions in scaled space
        scaler: Fitted MinMaxScaler
        n_features: Number of input features (default: 12 for optimal features)

    Returns:
        y_pred_original: Predictions in original price scale
    """
    # Scaler was fitted on [target, feature1, ..., featureN]
    # Total columns = 1 (target) + n_features
    total_cols = 1 + n_features

    # Create dummy array with predictions in first column (target position)
    y_pred_full = np.concatenate([
        y_pred_scaled.reshape(-1, 1),
        np.zeros((len(y_pred_scaled), n_features))
    ], axis=1)

    # Verify shape matches scaler
    if y_pred_full.shape[1] != total_cols:
        raise ValueError(f"Shape mismatch: y_pred_full has {y_pred_full.shape[1]} columns "
                        f"but scaler expects {total_cols} columns")

    # Inverse transform
    y_pred_unscaled = scaler.inverse_transform(y_pred_full)

    # Inverse log transform (expm1)
    y_pred_original = np.expm1(y_pred_unscaled[:, 0])

    return y_pred_original
