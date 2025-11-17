"""
FastAPI web service for electricity consumption forecasting.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime, timedelta
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(
    title="Electricity Consumption Forecasting API",
    description="Predict next-day household energy consumption",
    version="1.0.0"
)

# Load model and feature info
MODEL_DIR = Path('models')
MODEL_PATH = MODEL_DIR / 'electricity_forecast_model.joblib'
FEATURE_PATH = MODEL_DIR / 'model_features.json'

model = None
feature_columns = None


def load_model():
    """Load the trained model and feature configuration."""
    global model, feature_columns
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please train the model first.")
    
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(f"Feature info not found at {FEATURE_PATH}. Please train the model first.")
    
    model = joblib.load(MODEL_PATH)
    
    with open(FEATURE_PATH, 'r') as f:
        feature_info = json.load(f)
        feature_columns = feature_info['feature_columns']
    
    print(f"Model loaded successfully. Features: {len(feature_columns)}")


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    try:
        load_model()
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        print("Model will need to be loaded before making predictions.")


class PredictionRequest(BaseModel):
    """Request model for prediction."""
    # Historical daily consumption values (kWh)
    historical_consumption: List[float]
    # Optional: specific date to predict for (defaults to tomorrow)
    target_date: Optional[str] = None


class PredictionResponse(BaseModel):
    """Response model for prediction."""
    predicted_consumption_kwh: float
    target_date: str
    model_type: str
    confidence_interval_95: Optional[dict] = None


def create_features_from_history(historical: list[float], target_date: datetime) -> pd.DataFrame:
    """
    Create feature vector from historical consumption data.
    
    Args:
        historical: List of daily consumption values (most recent last)
        target_date: Date to predict for
    
    Returns:
        DataFrame with single row of features
    """
    # Reverse to have oldest first
    hist = list(reversed(historical))
    
    # Create a temporary dataframe to compute features
    # We need at least 30 days for rolling features
    if len(hist) < 30:
        # Pad with mean if needed
        mean_val = np.mean(hist)
        hist = [mean_val] * (30 - len(hist)) + hist
    
    # Create date range ending at target_date - 1 day
    dates = pd.date_range(end=target_date - timedelta(days=1), periods=len(hist), freq='D')
    temp_df = pd.DataFrame({'kwh': hist}, index=dates)
    
    # Create features using the same logic as training
    features = temp_df.copy()
    
    # Lag features
    for lag in [1, 2, 3, 7, 14]:
        if len(features) >= lag:
            features[f'lag_{lag}'] = features['kwh'].shift(lag)
        else:
            features[f'lag_{lag}'] = np.nan
    
    # Rolling statistics
    for window in [7, 14, 30]:
        features[f'rolling_mean_{window}'] = features['kwh'].rolling(window=window, min_periods=1).mean()
        features[f'rolling_std_{window}'] = features['kwh'].rolling(window=window, min_periods=1).std()
    
    # Calendar features for target date
    features['day_of_week'] = target_date.dayofweek
    features['month'] = target_date.month
    features['day_of_month'] = target_date.day
    features['is_weekend'] = 1 if target_date.dayofweek >= 5 else 0
    
    # Fourier terms
    features['sin_week'] = np.sin(2 * np.pi * target_date.dayofweek / 7)
    features['cos_week'] = np.cos(2 * np.pi * target_date.dayofweek / 7)
    features['sin_month'] = np.sin(2 * np.pi * target_date.month / 12)
    features['cos_month'] = np.cos(2 * np.pi * target_date.month / 12)
    
    # Get the last row (most recent features)
    feature_row = features.iloc[-1][feature_columns].to_frame().T
    
    # Fill any remaining NaN with 0
    feature_row = feature_row.fillna(0)
    
    return feature_row


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "Electricity Consumption Forecasting API",
        "status": "running",
        "model_loaded": model is not None
    }


@app.get("/health")
async def health():
    """Health check with model status."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH) if MODEL_PATH.exists() else None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict next-day electricity consumption.
    
    Requires historical consumption data (at least 14 days recommended).
    """
    if model is None:
        try:
            load_model()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model not available: {str(e)}")
    
    if len(request.historical_consumption) < 14:
        raise HTTPException(
            status_code=400,
            detail="At least 14 days of historical consumption data required"
        )
    
    # Parse target date
    if request.target_date:
        try:
            target_date = pd.to_datetime(request.target_date)
        except:
            raise HTTPException(status_code=400, detail="Invalid target_date format. Use YYYY-MM-DD")
    else:
        target_date = datetime.now() + timedelta(days=1)
    
    # Create features
    try:
        feature_row = create_features_from_history(
            request.historical_consumption,
            target_date
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature creation failed: {str(e)}")
    
    # Make prediction
    try:
        prediction = model.predict(feature_row)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    # Get model type
    model_type = type(model).__name__
    
    return PredictionResponse(
        predicted_consumption_kwh=float(prediction),
        target_date=target_date.strftime("%Y-%m-%d"),
        model_type=model_type
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

