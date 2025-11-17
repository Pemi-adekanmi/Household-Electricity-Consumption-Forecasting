"""
Training script for electricity consumption forecasting model.
Exported from modeling notebook for production use.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
import argparse


def load_data(data_path):
    """Load and preprocess the household power consumption data."""
    dtype_spec = {
        'Date': 'string',
        'Time': 'string',
        'Global_active_power': 'string',
        'Global_reactive_power': 'string',
        'Voltage': 'string',
        'Global_intensity': 'string',
        'Sub_metering_1': 'string',
        'Sub_metering_2': 'string',
        'Sub_metering_3': 'string'
    }
    
    df = pd.read_csv(
        data_path,
        sep=';',
        dtype=dtype_spec,
        na_values='?',
        parse_dates={'timestamp': ['Date', 'Time']},
        dayfirst=True,
        low_memory=False
    )
    
    numeric_cols = [
        'Global_active_power',
        'Global_reactive_power',
        'Voltage',
        'Global_intensity',
        'Sub_metering_1',
        'Sub_metering_2',
        'Sub_metering_3'
    ]
    
    df[numeric_cols] = df[numeric_cols].astype(float)
    df = df.dropna(subset=['Global_active_power']).reset_index(drop=True)
    df = df.sort_values('timestamp').set_index('timestamp')
    df = df[~df.index.duplicated(keep='first')]
    
    # Create daily energy target
    df['energy_kwh'] = df['Global_active_power'] / 60.0
    daily_energy = df['energy_kwh'].resample('D').sum().to_frame(name='kwh')
    daily_energy = daily_energy[daily_energy['kwh'] > 0]
    
    return daily_energy


def create_features(df, target_col='kwh'):
    """Create time-series features for forecasting."""
    features = df.copy()
    
    # Lag features (previous days)
    for lag in [1, 2, 3, 7, 14]:
        features[f'lag_{lag}'] = features[target_col].shift(lag)
    
    # Rolling statistics
    for window in [7, 14, 30]:
        features[f'rolling_mean_{window}'] = features[target_col].rolling(window=window, min_periods=1).mean()
        features[f'rolling_std_{window}'] = features[target_col].rolling(window=window, min_periods=1).std()
    
    # Calendar features
    features['day_of_week'] = features.index.dayofweek
    features['month'] = features.index.month
    features['day_of_month'] = features.index.day
    features['is_weekend'] = (features.index.dayofweek >= 5).astype(int)
    
    # Fourier terms for seasonality
    features['sin_week'] = np.sin(2 * np.pi * features.index.dayofweek / 7)
    features['cos_week'] = np.cos(2 * np.pi * features.index.dayofweek / 7)
    features['sin_month'] = np.sin(2 * np.pi * features.index.month / 12)
    features['cos_month'] = np.cos(2 * np.pi * features.index.month / 12)
    
    return features


def train_model(X_train, y_train, X_val, y_val, tune=True):
    """Train and optionally tune a Random Forest model."""
    if tune:
        print("Tuning Random Forest hyperparameters...")
        rf_param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [8, 10, 12],
            'min_samples_split': [2, 5]
        }
        
        tscv = TimeSeriesSplit(n_splits=3)
        rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
        rf_grid = GridSearchCV(
            rf_base, rf_param_grid, cv=tscv, 
            scoring='neg_mean_squared_error', n_jobs=-1, verbose=1
        )
        rf_grid.fit(X_train, y_train)
        
        print(f"Best parameters: {rf_grid.best_params_}")
        model = rf_grid.best_estimator_
    else:
        print("Training Random Forest with default parameters...")
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
    
    # Evaluate
    pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, pred))
    mae = mean_absolute_error(y_val, pred)
    r2 = r2_score(y_val, pred)
    
    print(f"\nModel Performance:")
    print(f"  RMSE: {rmse:.2f} kWh")
    print(f"  MAE: {mae:.2f} kWh")
    print(f"  R²: {r2:.3f}")
    
    return model, {'rmse': rmse, 'mae': mae, 'r2': r2}


def main():
    parser = argparse.ArgumentParser(description='Train electricity consumption forecasting model')
    parser.add_argument('--data', type=str, default='data/household_power_consumption.txt',
                        help='Path to input data file')
    parser.add_argument('--output', type=str, default='models',
                        help='Output directory for model artifacts')
    parser.add_argument('--tune', action='store_true',
                        help='Perform hyperparameter tuning (slower)')
    parser.add_argument('--train-split', type=float, default=0.8,
                        help='Fraction of data to use for training')
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    daily_energy = load_data(data_path)
    print(f"Loaded {len(daily_energy)} daily records")
    
    # Create features
    print("Creating features...")
    features_df = create_features(daily_energy)
    features_df = features_df.dropna().copy()
    print(f"Features shape: {features_df.shape}")
    
    # Split data
    split_idx = int(len(features_df) * args.train_split)
    train = features_df.iloc[:split_idx].copy()
    val = features_df.iloc[split_idx:].copy()
    
    print(f"\nTrain: {len(train)} days ({train.index.min()} to {train.index.max()})")
    print(f"Validation: {len(val)} days ({val.index.min()} to {val.index.max()})")
    
    # Prepare features and target
    feature_cols = [c for c in features_df.columns if c != 'kwh']
    X_train = train[feature_cols]
    y_train = train['kwh']
    X_val = val[feature_cols]
    y_val = val['kwh']
    
    # Train model
    model, metrics = train_model(X_train, y_train, X_val, y_val, tune=args.tune)
    
    # Save model
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    model_path = output_dir / 'electricity_forecast_model.joblib'
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Save feature info
    feature_info = {
        'feature_columns': feature_cols,
        'target_column': 'kwh',
        'model_type': 'RandomForestRegressor',
        'metrics': metrics,
        'train_size': len(train),
        'val_size': len(val)
    }
    
    feature_path = output_dir / 'model_features.json'
    with open(feature_path, 'w') as f:
        json.dump(feature_info, f, indent=2, default=str)
    print(f"Feature info saved to: {feature_path}")


if __name__ == '__main__':
    main()

