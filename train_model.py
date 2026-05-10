"""
Train Air Pollution Prediction Model
Predicts PM2.5, PM10 levels and pollution range
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report
import xgboost as xgb
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load pollution dataset"""
    df = pd.read_csv('data/pollution_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    return df

def create_features(df):
    """Create time-based features"""
    df = df.copy()
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['season'] = df['month'].apply(lambda x: 'dry' if x in [11,12,1,2,3,4] else 'wet')
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Encode categorical variables
    le_wind = LabelEncoder()
    df['wind_dir_encoded'] = le_wind.fit_transform(df['wind_direction'])
    
    le_season = LabelEncoder()
    df['season_encoded'] = le_season.fit_transform(df['season'])
    
    return df, le_wind, le_season

def prepare_features_regression(df):
    """Prepare features for regression (PM2.5 prediction)"""
    feature_cols = [
        'day_of_week', 'month', 'is_weekend', 'wind_speed_kmh',
        'wind_dir_encoded', 'season_encoded', 'temperature_c', 'humidity_pct',
        'mining_active'
    ]
    
    X = df[feature_cols].values
    y_pm25 = df['pm25'].values
    y_pm10 = df['pm10'].values
    
    return X, y_pm25, y_pm10, feature_cols

def prepare_features_classification(df):
    """Prepare features for classification (AQI category)"""
    feature_cols = [
        'day_of_week', 'month', 'is_weekend', 'wind_speed_kmh',
        'wind_dir_encoded', 'season_encoded', 'temperature_c', 'humidity_pct',
        'mining_active'
    ]
    
    X = df[feature_cols].values
    y = df['aqi_category'].values
    
    # Encode target
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    
    return X, y_encoded, le_target, feature_cols

def train_regression_models(X_train, X_test, y_train, y_test, target_name):
    """Train regression models for pollution prediction"""
    
    print(f"\n{'='*50}")
    print(f"Training {target_name} Prediction Models")
    print(f"{'='*50}")
    
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = model.score(X_test, y_test)
        
        results[name] = {
            'model': model,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'predictions': predictions
        }
        
        print(f"\n{name}:")
        print(f"  MAE: {mae:.2f} µg/m³")
        print(f"  RMSE: {rmse:.2f} µg/m³")
        print(f"  R² Score: {r2:.3f}")
    
    return results

def train_classification_model(X_train, X_test, y_train, y_test, target_encoder):
    """Train classification model for AQI category"""
    
    print(f"\n{'='*50}")
    print("Training AQI Category Classification Model")
    print(f"{'='*50}")
    
    model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"\nXGBoost Classifier:")
    print(f"  Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=target_encoder.classes_))
    
    return model

def predict_pollution_range(pm25_value):
    """Predict pollution range based on PM2.5 value"""
    
    if pm25_value <= 12:
        return {'range': '0-12', 'level': 'Good', 'color': 'Green'}
    elif pm25_value <= 35.4:
        return {'range': '12-35', 'level': 'Moderate', 'color': 'Yellow'}
    elif pm25_value <= 55.4:
        return {'range': '35-55', 'level': 'Unhealthy for Sensitive Groups', 'color': 'Orange'}
    elif pm25_value <= 150.4:
        return {'range': '55-150', 'level': 'Unhealthy', 'color': 'Red'}
    elif pm25_value <= 250.4:
        return {'range': '150-250', 'level': 'Very Unhealthy', 'color': 'Purple'}
    else:
        return {'range': '250+', 'level': 'Hazardous', 'color': 'Maroon'}

def main():
    print("="*60)
    print("🌬️ AIR POLLUTION PREDICTION MODEL TRAINING")
    print("For Surigao del Sur")
    print("="*60)
    
    # Load data
    print("\n📂 Loading data...")
    df = load_data()
    print(f"Loaded {len(df):,} records")
    
    # Create features
    print("\n🔧 Creating features...")
    df, le_wind, le_season = create_features(df)
    
    # Prepare for regression
    X, y_pm25, y_pm10, feature_cols = prepare_features_regression(df)
    
    # Split data
    X_train, X_test, y_pm25_train, y_pm25_test, y_pm10_train, y_pm10_test = train_test_split(
        X, y_pm25, y_pm10, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train PM2.5 prediction models
    pm25_results = train_regression_models(X_train_scaled, X_test_scaled, y_pm25_train, y_pm25_test, "PM2.5")
    
    # Train PM10 prediction models
    pm10_results = train_regression_models(X_train_scaled, X_test_scaled, y_pm10_train, y_pm10_test, "PM10")
    
    # Prepare for classification
    X_class, y_class, le_target, class_features = prepare_features_classification(df)
    X_class_train, X_class_test, y_class_train, y_class_test = train_test_split(
        X_class, y_class, test_size=0.2, random_state=42
    )
    X_class_train_scaled = scaler.transform(X_class_train)
    X_class_test_scaled = scaler.transform(X_class_test)
    
    # Train classification model
    classifier = train_classification_model(X_class_train_scaled, X_class_test_scaled, y_class_train, y_class_test, le_target)
    
    # Save models
    os.makedirs('models', exist_ok=True)
    
    best_pm25_model = pm25_results['XGBoost']['model']
    best_pm10_model = pm10_results['XGBoost']['model']
    
    joblib.dump(best_pm25_model, 'models/pm25_predictor.pkl')
    joblib.dump(best_pm10_model, 'models/pm10_predictor.pkl')
    joblib.dump(classifier, 'models/aqi_classifier.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(le_wind, 'models/wind_encoder.pkl')
    joblib.dump(le_season, 'models/season_encoder.pkl')
    joblib.dump(le_target, 'models/aqi_encoder.pkl')
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print("\n📁 Models saved to:")
    print("   - models/pm25_predictor.pkl")
    print("   - models/pm10_predictor.pkl")
    print("   - models/aqi_classifier.pkl")
    print("   - models/scaler.pkl")
    
    # Example prediction
    print("\n🔮 Example Prediction:")
    example_features = np.array([[2, 3, 0, 12, 0, 1, 28, 75, 1]])  # [day, month, weekend, wind_speed, wind_dir, season, temp, humidity, mining]
    example_scaled = scaler.transform(example_features)
    
    pm25_pred = best_pm25_model.predict(example_scaled)[0]
    pm10_pred = best_pm10_model.predict(example_scaled)[0]
    pollution_range = predict_pollution_range(pm25_pred)
    
    print(f"   Predicted PM2.5: {pm25_pred:.1f} µg/m³")
    print(f"   Predicted PM10: {pm10_pred:.1f} µg/m³")
    print(f"   Pollution Range: {pollution_range['range']} µg/m³")
    print(f"   Air Quality Level: {pollution_range['level']}")
    
    print("\n🚀 Run: python app.py to start the dashboard")

if __name__ == "__main__":
    main()