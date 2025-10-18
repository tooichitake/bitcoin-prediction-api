# Bitcoin Price Prediction FastAPI

FastAPI application for predicting next-day high price of Bitcoin using Linear Regression.

## 🎯 Project Overview

This API is part of **AT3 Assignment** for **36120 Advanced Machine Learning** at UTS. It provides real-time Bitcoin price prediction using a trained Linear Regression model.

## 📋 Features

- **Real-time Data Fetching**: Automatically fetches Bitcoin data from CryptoCompare API
- **Price Prediction**: Predicts next-day high price using Linear Regression
- **RESTful API**: Three endpoints following assignment requirements
- **Docker Support**: Containerized for easy deployment
- **Production Ready**: Health checks, error handling, and proper documentation

## 🚀 API Endpoints

### 1. `GET /`
**Description**: Displaying a brief description of the project objectives, list of endpoints, expected input parameters and output format of the model, link to the Github repo.

**Response**: Project information and API documentation.

---

### 2. `GET /health/`
**Description**: Health check endpoint - returning status code 200 with a welcome message.

**Response Example:**
```json
{
  "status": "healthy",
  "message": "Bitcoin Price Prediction API is running",
  "timestamp": "2025-10-18T10:30:00.123456"
}
```

---

### 3. `GET /predict/{token}`
**Description**: Returning the prediction on the trained model (HIGH price of the token the next day).

#### Expected Input Parameters

| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `token` | Path | ✅ Yes | Cryptocurrency token symbol | `bitcoin` or `btc` |
| `date` | Query | ✅ Yes | Date from which the model will predict the next day's high price. Format: `YYYY-MM-DD` | `2025-10-18` |

#### Request Example
```bash
GET /predict/bitcoin?date=2025-10-18
```

#### Expected Output Format
```json
{
  "input_date": "2025-10-18",
  "prediction": {
    "target_date": "2025-10-19",
    "predicted_high_price": 67234.56
  }
}
```

#### Output Fields

| Field | Type | Description |
|-------|------|-------------|
| `input_date` | string | Date of the data used for prediction (format: YYYY-MM-DD) |
| `prediction.target_date` | string | Date for which the prediction is made (input_date + 1 day) |
| `prediction.predicted_high_price` | float | Predicted high price in USD |

## 🛠️ Tech Stack

- **Python**: 3.11.4
- **FastAPI**: 0.111.0
- **Uvicorn**: 0.30.1
- **scikit-learn**: 1.5.1
- **Pandas**: 2.2.2
- **Docker**: For containerization

## 📦 Installation

### Local Setup

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd fastAPI
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the API**
```bash
uvicorn app.main:app --reload
```

5. **Access the API**
- API: http://localhost:8000
- Interactive Docs: http://localhost:8000/docs
- Alternative Docs: http://localhost:8000/redoc

### Docker Setup

1. **Build Docker image**
```bash
docker build -t bitcoin-prediction-api .
```

2. **Run container**
```bash
docker run -p 8000:8000 bitcoin-prediction-api
```

3. **Access the API**
- API: http://localhost:8000

## 📁 Project Structure

```
fastAPI/
├── app/
│   └── main.py              # FastAPI application
├── models/
│   └── linear_regression_bitcoin.pkl  # Trained model
├── Dockerfile               # Docker configuration
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── github.txt              # GitHub repository link
```

## 🧪 Testing

### Test Health Endpoint
```bash
curl http://localhost:8000/health/
```

### Test Root Endpoint
```bash
curl http://localhost:8000/
```

### Test Prediction Endpoint
```bash
# Example: Predict high price for 2025-10-19 based on 2025-10-18 data
curl "http://localhost:8000/predict/bitcoin?date=2025-10-18"
```

**Expected Response:**
```json
{
  "input_date": "2025-10-18",
  "prediction": {
    "target_date": "2025-10-19",
    "predicted_high_price": 67234.56
  }
}
```

## 🌐 Deployment

### Deploy to Render

1. Push code to GitHub
2. Create new Web Service on [Render](https://render.com)
3. Connect GitHub repository
4. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Environment**: Python 3.11.4
5. Deploy

### Environment Variables (Optional)
- `PORT`: API port (default: 8000)

## 📊 Model Information

- **Algorithm**: Linear Regression (sklearn)
- **Features**: 30 optimal features selected from correlation analysis
- **Feature Types**: Technical indicators (SMA, EMA, DEMA, TEMA, HMA, VWAP, Bollinger Bands, Keltner Channels, Donchian Channels, MACD)
- **Target**: Next-day high price
- **Training Data**: Bitcoin historical data
- **Model File**: `models/linear_regression_bitcoin.pkl`
- **Preprocessing**: Raw features (no transformation)

## 🔗 Data Sources

- **CryptoCompare API**: Real-time and historical Bitcoin data (free tier)
- No API key required for basic usage
- Rate limits apply (as per CryptoCompare free tier)

## 📝 Assignment Requirements

✅ **Git Repository Structure**
- `app/`: FastAPI application code
- `models/`: Trained model artifacts
- `Dockerfile`: Docker configuration
- `requirements.txt`: Python dependencies
- `github.txt`: Repository link

✅ **Required Endpoints**
- `GET /`: Documentation
- `GET /health/`: Health check
- `GET /predict/{token}`: Prediction endpoint

✅ **Technologies**
- Python 3.11.4
- FastAPI 0.111.0
- Uvicorn 0.30.1
- scikit-learn 1.5.1

## 🐛 Troubleshooting

### Model Not Loading
- Ensure `models/linear_regression_bitcoin.pkl` exists
- Check file permissions
- Verify scikit-learn version (1.5.1)

### API Connection Errors
- Check if CryptoCompare API is accessible
- Verify internet connection
- Check rate limits

### Docker Issues
- Ensure Docker daemon is running
- Check port 8000 is not in use
- Verify all files are copied correctly

## 👥 Contributors

- **Group 4** - AT3 Assignment
- **Course**: 36120 Advanced Machine Learning
- **Institution**: University of Technology Sydney (UTS)

## 📄 License

This project is for educational purposes as part of UTS coursework.

## 🔗 Links

- **API Repository**: https://github.com/tooichitake/bitcoin-prediction-api
- **Interactive API Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
