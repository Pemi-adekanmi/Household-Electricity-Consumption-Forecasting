# Household Electricity Consumption Forecasting

Midterm project for ML Zoomcamp: Predicting next-day household energy consumption using machine learning.

## Project Overview

This project forecasts daily household electricity consumption (kWh) using historical consumption patterns, calendar features, and time-series modeling techniques.

### Problem Statement

Energy consumption forecasting helps households and utilities:
- **Demand planning**: Anticipate energy needs to optimize usage
- **Cost management**: Reduce peak-hour consumption to lower bills
- **Grid stability**: Enable better load balancing for power grids
- **Smart home automation**: Automatically adjust devices based on predicted demand

### Dataset

UCI Machine Learning Repository: [Individual household electric power consumption](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)

- **Period**: December 2006 - November 2010 (4 years)
- **Frequency**: Minute-level measurements
- **Features**: Global active/reactive power, voltage, current, sub-metering (kitchen, laundry, water-heater/AC)
- **Target**: Daily aggregated energy consumption (kWh)

## Project Structure

```
.
├── data/                           # Dataset files
│   └── household_power_consumption.txt
├── notebooks/                      # Jupyter notebooks
│   ├── 01_power_consumption_eda.ipynb
│   └── 02_power_consumption_modeling.ipynb
├── models/                         # Trained model artifacts (created after training)
│   ├── electricity_forecast_model.joblib
│   └── model_features.json
├── train.py                        # Training script
├── predict.py                      # FastAPI web service
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker configuration
└── README.md                       # This file
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

The dataset should be in `data/household_power_consumption.txt`. If not, download from:
https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip

Extract and place `household_power_consumption.txt` in the `data/` directory.

## Usage

### Training the Model

Train the model using the training script:

```bash
# Basic training (fast)
python train.py

# With hyperparameter tuning (slower but better performance)
python train.py --tune

# Custom data path and output directory
python train.py --data data/household_power_consumption.txt --output models --tune
```

The script will:
1. Load and preprocess the data
2. Create time-series features (lags, rolling stats, calendar features)
3. Split data chronologically (80% train, 20% validation)
4. Train a Random Forest model (with optional tuning)
5. Save the model and feature configuration to `models/`

### Running the Web Service

Start the FastAPI service:

```bash
python predict.py
```

Or using uvicorn directly:

```bash
uvicorn predict:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

#### API Endpoints

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Make Prediction:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "historical_consumption": [15.2, 16.1, 14.8, 17.3, 15.9, 18.2, 16.5, 15.7, 17.1, 16.8, 15.4, 18.0, 16.3, 17.5],
    "target_date": "2024-01-15"
  }'
```

Response:
```json
{
  "predicted_consumption_kwh": 16.8,
  "target_date": "2024-01-15",
  "model_type": "RandomForestRegressor"
}
```

### Docker Deployment

Build the Docker image:

```bash
docker build -t electricity-forecast .
```

Run the container:

```bash
# Mount models directory if you've already trained
docker run -p 8000:8000 -v $(pwd)/models:/app/models electricity-forecast

# Or train inside container first
docker run -p 8000:8000 -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models electricity-forecast python train.py
docker run -p 8000:8000 -v $(pwd)/models:/app/models electricity-forecast
```

## Methodology

### Data Preparation

1. **Loading**: Parse semicolon-delimited CSV with proper date/time handling
2. **Cleaning**: Handle missing values (`?`), remove duplicates, filter invalid readings
3. **Aggregation**: Convert minute-level power (kW) to daily energy (kWh)
4. **Feature Engineering**:
   - **Lag features**: Previous 1, 2, 3, 7, 14 days
   - **Rolling statistics**: 7, 14, 30-day moving averages and standard deviations
   - **Calendar features**: Day of week, month, day of month, weekend indicator
   - **Fourier terms**: Cyclical encoding for weekly and monthly seasonality

### Modeling

- **Baseline**: Naive forecast (yesterday's value)
- **Models tested**: Linear Regression, Ridge, Random Forest, XGBoost
- **Best model**: Random Forest (tuned)
- **Validation**: Chronological split (80/20) to respect time-series nature
- **Metrics**: RMSE, MAE, R²

### Model Performance

Typical performance on validation set:
- **RMSE**: ~2.5-3.0 kWh/day
- **MAE**: ~1.8-2.2 kWh/day
- **R²**: ~0.75-0.85

## Key Insights

1. **Weekly patterns**: Weekends show different consumption patterns than weekdays
2. **Seasonal variation**: Higher consumption in winter (heating) and summer (cooling)
3. **Feature importance**: Lag-1 (yesterday) and rolling averages are most predictive
4. **Model selection**: Tree-based models (Random Forest) outperform linear models due to non-linear patterns

## Next Steps / Improvements

- [ ] Add external features (temperature, holidays)
- [ ] Experiment with LSTM/Transformer models for sequence learning
- [ ] Implement prediction intervals/uncertainty quantification
- [ ] Add model monitoring and retraining pipeline
- [ ] Deploy to cloud (AWS/GCP/Azure) with CI/CD

## License

This project is for educational purposes as part of ML Zoomcamp coursework.

## References

- UCI Machine Learning Repository: Household Power Consumption Dataset
- ML Zoomcamp Course Materials

