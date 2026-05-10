"""
Air Pollution Prediction Dashboard - Surigao del Sur
Real-time pollution monitoring and forecasting
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import random

app = Flask(__name__)

# Load models
try:
    pm25_model = joblib.load('models/pm25_predictor.pkl')
    pm10_model = joblib.load('models/pm10_predictor.pkl')
    aqi_classifier = joblib.load('models/aqi_classifier.pkl')
    scaler = joblib.load('models/scaler.pkl')
    wind_encoder = joblib.load('models/wind_encoder.pkl')
    season_encoder = joblib.load('models/season_encoder.pkl')
    aqi_encoder = joblib.load('models/aqi_encoder.pkl')
    models_loaded = True
    print("✅ Models loaded successfully")
except Exception as e:
    models_loaded = False
    print(f"⚠️ Models not loaded: {e}")
    print("Run python train_model.py first")

# Municipalities with mining activity
MUNICIPALITIES = {
    'Carrascal': {'mining_active': True, 'population': 25000},
    'Cantilan': {'mining_active': True, 'population': 34000},
    'Madrid': {'mining_active': True, 'population': 16000},
    'Lanuza': {'mining_active': False, 'population': 12000},
    'Tandag City': {'mining_active': False, 'population': 58000},
    'Hinatuan': {'mining_active': False, 'population': 42000},
    'Lianga': {'mining_active': False, 'population': 18000},
    'Tagbina': {'mining_active': False, 'population': 28000},
    'Barobo': {'mining_active': False, 'population': 23000},
    'Bislig City': {'mining_active': False, 'population': 96000}
}

def predict_pollution_range(pm25_value):
    """Get pollution range category"""
    if pm25_value <= 12:
        return {'range': '0-12', 'level': 'Good', 'color': '#4caf50', 'advice': 'Air quality is good. Safe for all activities.'}
    elif pm25_value <= 35.4:
        return {'range': '12-35', 'level': 'Moderate', 'color': '#ffc107', 'advice': 'Acceptable air quality. Unusually sensitive people should limit outdoor activity.'}
    elif pm25_value <= 55.4:
        return {'range': '35-55', 'level': 'Unhealthy for Sensitive Groups', 'color': '#ff9800', 'advice': 'Active children and adults should limit outdoor activity.'}
    elif pm25_value <= 150.4:
        return {'range': '55-150', 'level': 'Unhealthy', 'color': '#f44336', 'advice': 'Everyone should limit outdoor activity. Wear masks if going outside.'}
    elif pm25_value <= 250.4:
        return {'range': '150-250', 'level': 'Very Unhealthy', 'color': '#9c27b0', 'advice': 'Health alert. Stay indoors. Use air purifier if available.'}
    else:
        return {'range': '250+', 'level': 'Hazardous', 'color': '#795548', 'advice': 'Health warning. Emergency conditions. Stay indoors at all times.'}

def get_health_advisory(pm25, location, wind_speed, wind_dir):
    """Generate health advisory based on pollution levels"""
    
    if pm25 <= 50:
        return {
            'risk': 'Low',
            'message': 'Air quality is good. Enjoy outdoor activities.',
            'precautions': ['No special precautions needed', 'Continue regular outdoor activities']
        }
    elif pm25 <= 100:
        return {
            'risk': 'Moderate',
            'message': 'Moderate air pollution. Sensitive groups should reduce outdoor time.',
            'precautions': ['Limit prolonged outdoor activity', 'Keep windows closed during peak hours', 'Wear mask if sensitive to pollution']
        }
    elif pm25 <= 150:
        return {
            'risk': 'High',
            'message': 'Unhealthy air quality. Everyone should limit outdoor exposure.',
            'precautions': ['Avoid outdoor exercise', 'Keep windows and doors closed', 'Use N95 mask when going out', 'Use air purifier indoors']
        }
    else:
        return {
            'risk': 'Critical',
            'message': 'Very unhealthy air quality. Health alert!',
            'precautions': ['Stay indoors as much as possible', 'Close all windows and doors', 'Wear N95 mask if you must go out', 'Seek medical help if experiencing breathing difficulties', 'Check on elderly and children']
        }

@app.route('/')
def index():
    return render_template('pollution_dashboard.html', municipalities=MUNICIPALITIES)

@app.route('/predict', methods=['POST'])
def predict():
    """Predict pollution for selected municipality"""
    
    data = request.get_json()
    municipality = data.get('municipality')
    
    if municipality not in MUNICIPALITIES:
        return jsonify({'error': 'Municipality not found'}), 404
    
    info = MUNICIPALITIES[municipality]
    
    # Get current weather data (in real system, fetch from API)
    current_month = datetime.now().month
    is_dry = current_month in [11, 12, 1, 2, 3, 4]
    
    # Simulate current wind conditions
    if current_month in [11, 12, 1, 2, 3]:
        wind_dir = 'NE'
        wind_speed = random.uniform(8, 15)
    elif current_month in [6, 7, 8, 9, 10]:
        wind_dir = 'SW'
        wind_speed = random.uniform(10, 20)
    else:
        wind_dir = 'E'
        wind_speed = random.uniform(5, 12)
    
    # Prepare features for prediction
    current_day = datetime.now().weekday()
    is_weekend = 1 if current_day >= 5 else 0
    season = 'dry' if is_dry else 'wet'
    
    # Encode categorical features
    try:
        wind_encoded = wind_encoder.transform([wind_dir])[0]
        season_encoded = season_encoder.transform([season])[0]
    except:
        wind_encoded = 0
        season_encoded = 0
    
    mining_active = 1 if info['mining_active'] else 0
    
    # Create feature vector
    features = np.array([[
        current_day,          # day_of_week
        current_month,        # month
        is_weekend,          # is_weekend
        wind_speed,          # wind_speed_kmh
        wind_encoded,        # wind_dir_encoded
        season_encoded,      # season_encoded
        random.uniform(24, 32),  # temperature_c
        random.uniform(70, 90),  # humidity_pct
        mining_active        # mining_active
    ]])
    
    if models_loaded:
        features_scaled = scaler.transform(features)
        pm25_pred = pm25_model.predict(features_scaled)[0]
        pm10_pred = pm10_model.predict(features_scaled)[0]
        aqi_pred = aqi_classifier.predict(features_scaled)[0]
        aqi_category = aqi_encoder.inverse_transform([aqi_pred])[0]
    else:
        # Demo mode
        if info['mining_active']:
            pm25_pred = random.uniform(50, 120)
        else:
            pm25_pred = random.uniform(20, 60)
        pm10_pred = pm25_pred * random.uniform(1.2, 1.8)
        aqi_category = 'Moderate' if pm25_pred < 50 else 'Unhealthy' if pm25_pred < 100 else 'Very Unhealthy'
    
    # Get pollution range
    range_info = predict_pollution_range(pm25_pred)
    
    # Get health advisory
    health = get_health_advisory(pm25_pred, municipality, wind_speed, wind_dir)
    
    # Generate 3-day forecast
    forecast = []
    for day in range(1, 4):
        forecast_date = datetime.now() + timedelta(days=day)
        
        # Simple forecast: similar to current with some variation
        forecast_pm25 = pm25_pred * random.uniform(0.8, 1.2)
        forecast_range = predict_pollution_range(forecast_pm25)
        
        forecast.append({
            'day': forecast_date.strftime('%A'),
            'date': forecast_date.strftime('%b %d'),
            'pm25': round(forecast_pm25, 1),
            'level': forecast_range['level'],
            'color': forecast_range['color']
        })
    
    # Wind impact analysis
    wind_analysis = ""
    if wind_speed > 15:
        wind_analysis = "High wind speeds are dispersing pollutants."
    elif wind_speed > 8:
        wind_analysis = "Moderate winds are carrying pollutants."
    else:
        wind_analysis = "Low wind speeds may cause pollutant accumulation."
    
    if info['mining_active']:
        wind_analysis += " Mining operations in this area contribute to particulate pollution."
    
    return jsonify({
        'municipality': municipality,
        'mining_active': info['mining_active'],
        'pm25': round(pm25_pred, 1),
        'pm10': round(pm10_pred, 1),
        'aqi_category': aqi_category,
        'pollution_range': range_info['range'],
        'pollution_level': range_info['level'],
        'range_color': range_info['color'],
        'range_advice': range_info['advice'],
        'wind_speed': round(wind_speed, 1),
        'wind_direction': wind_dir,
        'temperature': round(features[0][6], 1),
        'humidity': round(features[0][7], 1),
        'health_risk': health['risk'],
        'health_message': health['message'],
        'precautions': health['precautions'],
        'wind_analysis': wind_analysis,
        'forecast': forecast,
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/historical/<municipality>')
def historical_data(municipality):
    """Get historical pollution data for chart"""
    
    # Generate sample historical data for the last 30 days
    dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(30, 0, -1)]
    
    info = MUNICIPALITIES.get(municipality, {'mining_active': False})
    
    pm25_values = []
    pm10_values = []
    
    for i in range(30):
        if info['mining_active']:
            base = 60
            variation = random.uniform(-30, 30)
        else:
            base = 30
            variation = random.uniform(-15, 25)
        
        pm25 = max(5, base + variation)
        pm25_values.append(round(pm25, 1))
        pm10_values.append(round(pm25 * random.uniform(1.2, 1.6), 1))
    
    return jsonify({
        'dates': dates,
        'pm25': pm25_values,
        'pm10': pm10_values
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)