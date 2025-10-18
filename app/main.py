"""
Bitcoin Price Prediction API - AT3 Assignment
Predict next-day high price using Linear Regression with 30 optimal features
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import joblib
import pandas as pd
import pandas_ta_classic as ta  # Register .ta accessor
from datetime import datetime, timedelta
from pathlib import Path
import requests
import warnings
warnings.filterwarnings('ignore')

# Initialize FastAPI
app = FastAPI(
    title="Bitcoin Price Prediction API",
    description="Predict next-day high price for Bitcoin using Linear Regression",
    version="1.0.0"
)

# Load model
MODEL_PATH = Path(__file__).parent.parent / "models" / "linear_regression_bitcoin.pkl"
model = joblib.load(MODEL_PATH)

# Feature columns (30 optimal features selected from Top-N performance analysis)
FEATURE_COLS = [
    'close', 'VWAP', 'high', 'low', 'marketCap', 'open',
    'TEMA_20', 'SMA_7', 'DEMA_20', 'HMA_20', 'EMA_12', 'KCBe_20_2.0',
    'DCM_20_20', 'KCUe_20_2.0', 'KCLe_20_2.0', 'DCU_20_20',
    'BBM_20_2.0', 'SMA_20', 'EMA_26', 'BBU_20_2.0',
    'DCL_20_20', 'BBL_20_2.0', 'EMA_50', 'SMA_50',
    'SMA_200', 'ATRr_14', 'AD', 'OBV', 'volume', 'MACDs_12_26_9'
]


def fetch_bitcoin_data(days: int = 250) -> pd.DataFrame:
    """Fetch Bitcoin data from CryptoCompare API"""
    response = requests.get(
        "https://min-api.cryptocompare.com/data/v2/histoday",
        params={
            'fsym': 'BTC',
            'tsym': 'USD',
            'limit': days,
            'toTs': int(datetime.now().timestamp())
        },
        timeout=15
    )
    response.raise_for_status()
    data = response.json()

    if data.get('Response') != 'Success':
        raise ValueError(data.get('Message', 'API error'))

    df = pd.DataFrame(data['Data']['Data'])
    df['date'] = pd.to_datetime(df['time'], unit='s')
    df = df.rename(columns={'volumefrom': 'volume'})
    df['marketCap'] = df['close'] * df['volume']

    return df[['date', 'open', 'high', 'low', 'close', 'volume', 'marketCap']].sort_values('date').reset_index(drop=True)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add 54 technical indicators matching training pipeline"""
    df = df.copy()

    # Trend indicators
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
    df_temp = df.set_index(pd.to_datetime(df['date']))
    df['VWAP'] = df_temp.ta.vwap().values

    # Momentum indicators
    df.ta.rsi(length=14, append=True)
    df.ta.rsi(length=21, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.stoch(k=14, d=3, smooth_k=3, append=True)
    df.ta.willr(length=14, append=True)
    df.ta.roc(length=10, append=True)
    df.ta.roc(length=21, append=True)
    df.ta.cci(length=14, append=True)
    df.ta.mfi(length=14, append=True)

    # Volatility indicators
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.natr(length=14, append=True)
    df.ta.kc(length=20, scalar=2, append=True)
    df.ta.donchian(lower_length=20, upper_length=20, append=True)

    # Volume indicators
    df.ta.obv(append=True)
    df.ta.ad(append=True)
    df.ta.adosc(fast=3, slow=10, append=True)
    df.ta.cmf(length=20, append=True)

    # Trend strength
    df.ta.adx(length=14, append=True)
    df.ta.aroon(length=25, append=True)

    # Custom features
    df['price_to_sma20'] = (df['close'] - df['SMA_20']) / df['SMA_20'] * 100
    df['price_to_sma50'] = (df['close'] - df['SMA_50']) / df['SMA_50'] * 100
    df['sma20_50_ratio'] = df['SMA_20'] / df['SMA_50']
    df['bb_width_pct'] = (df['BBU_20_2.0'] - df['BBL_20_2.0']) / df['BBM_20_2.0'] * 100
    df['bb_position'] = (df['close'] - df['BBL_20_2.0']) / (df['BBU_20_2.0'] - df['BBL_20_2.0'])
    df['volume_surge'] = df['volume'] / df['volume'].rolling(window=20).mean()
    df['price_momentum_5'] = df['close'].pct_change(5) * 100
    df['price_momentum_10'] = df['close'].pct_change(10) * 100

    return df


@app.get("/")
async def root():
    """API documentation with expected input and output format"""
    return {
        "project": "Bitcoin Price Prediction API",
        "objective": "Predict next-day high price for Bitcoin using Linear Regression with 30 optimal features",
        "student": "Group 4 - AT3 Assignment",
        "course": "36120 Advanced Machine Learning - UTS",
        "github_repo": "https://github.com/tooichitake/bitcoin-prediction-api",
        "endpoints": {
            "GET /": "API documentation - displaying project objectives, endpoints list, input/output format, and Github repo",
            "GET /health/": "Health check - returning status code 200 with welcome message",
            "GET /predict/{token}": "Predict next-day HIGH price for the specified token (bitcoin or btc)"
        },
        "model_info": {
            "algorithm": "Linear Regression",
            "n_features": 30,
            "preprocessing": "Raw features (no transformation)",
            "feature_selection": "Top-30 optimal features from correlation analysis"
        },
        "expected_input_parameters": {
            "token": {
                "type": "path parameter (required)",
                "description": "Cryptocurrency token symbol",
                "allowed_values": ["bitcoin", "btc"],
                "example": "/predict/bitcoin"
            },
            "date": {
                "type": "query parameter (required)",
                "description": "Date from which to predict the next day's high price. Format: YYYY-MM-DD",
                "example": "?date=2025-10-18"
            }
        },
        "expected_output_format": {
            "description": "JSON response with input date and prediction details",
            "example": {
                "input_date": "2025-10-18",
                "prediction": {
                    "target_date": "2025-10-19",
                    "predicted_high_price": 67234.56
                }
            }
        }
    }


@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "message": "Bitcoin Price Prediction API is running",
            "timestamp": datetime.now().isoformat()
        }
    )


@app.get("/predict/{token}")
async def predict_token(token: str, date: str):
    """
    Predict next-day high price for Bitcoin

    Args:
        token: Token symbol (bitcoin or btc)
        date: Date from which to predict the next day's high price (format: YYYY-MM-DD)

    Returns:
        JSON response with input date, prediction details, and metadata

    Example:
        GET /predict/bitcoin?date=2025-10-18
    """
    if token.lower() not in ['bitcoin', 'btc']:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported token '{token}'. Only 'bitcoin' or 'btc' supported."
        )

    try:
        # Validate and parse date
        try:
            requested_date = pd.to_datetime(date, format="%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid date format '{date}'. Expected format: YYYY-MM-DD (e.g., 2025-10-18)"
            )

        # Fetch data and add indicators
        df = fetch_bitcoin_data(days=250)
        df = add_technical_indicators(df)

        # Remove warmup rows (199 days for SMA_200)
        df_clean = df.iloc[199:].reset_index(drop=True)
        df_clean['date'] = pd.to_datetime(df_clean['date'])

        # Find the requested date in the dataset
        matching_rows = df_clean[df_clean['date'] == requested_date]

        if matching_rows.empty:
            # Find closest available date
            available_dates = df_clean['date'].dt.strftime("%Y-%m-%d").tolist()
            raise HTTPException(
                status_code=404,
                detail=f"No data available for date '{date}'. Available date range: {available_dates[0]} to {available_dates[-1]}"
            )

        input_date = requested_date
        X = df_clean[df_clean['date'] == requested_date][FEATURE_COLS].values

        # Make prediction
        pred_price = model.predict(X)[0]

        # Calculate target date (next day)
        target_date = input_date + timedelta(days=1)

        # Return simple dictionary response
        return {
            "input_date": input_date.strftime("%Y-%m-%d"),
            "prediction": {
                "target_date": target_date.strftime("%Y-%m-%d"),
                "predicted_high_price": round(float(pred_price), 2)
            }
        }

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to fetch Bitcoin data: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
