"""
Simple test script for the prediction API.
Run this after starting the API server.
"""

import requests
import json

API_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint."""
    response = requests.get(f"{API_URL}/health")
    print("Health Check:")
    print(json.dumps(response.json(), indent=2))
    print()

def test_predict():
    """Test prediction endpoint."""
    # Example: 14 days of historical consumption
    payload = {
        "historical_consumption": [
            15.2, 16.1, 14.8, 17.3, 15.9, 18.2, 16.5,
            15.7, 17.1, 16.8, 15.4, 18.0, 16.3, 17.5
        ],
        "target_date": "2024-01-15"
    }
    
    response = requests.post(f"{API_URL}/predict", json=payload)
    print("Prediction:")
    print(json.dumps(response.json(), indent=2))
    print()

if __name__ == "__main__":
    try:
        test_health()
        test_predict()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API. Make sure the server is running:")
        print("  python predict.py")
    except Exception as e:
        print(f"Error: {e}")

