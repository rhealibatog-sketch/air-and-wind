"""
Air Pollution and Wind Data Collection
For Surigao del Sur - Mining Areas and Coastal Communities
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import random

# Municipalities in Surigao del Sur with mining activity
MUNICIPALITIES = {
    'Carrascal': {'mining_active': True, 'coordinates': (9.3667, 125.9500)},
    'Cantilan': {'mining_active': True, 'coordinates': (9.3333, 125.9833)},
    'Madrid': {'mining_active': True, 'coordinates': (9.2667, 125.9667)},
    'Lanuza': {'mining_active': False, 'coordinates': (9.2333, 126.0500)},
    'Tandag City': {'mining_active': False, 'coordinates': (9.0833, 126.2000)},
    'Hinatuan': {'mining_active': False, 'coordinates': (8.3667, 126.3333)},
    'Lianga': {'mining_active': False, 'coordinates': (8.6333, 126.1000)},
    'Tagbina': {'mining_active': False, 'coordinates': (8.4500, 126.1667)},
    'Barobo': {'mining_active': False, 'coordinates': (8.5333, 126.2167)},
    'Bislig City': {'mining_active': False, 'coordinates': (8.2167, 126.3167)}
}

# Air Quality Standards (Philippine Clean Air Act)
AQI_LEVELS = {
    'Good': (0, 50, 'Green', 'Minimal impact'),
    'Moderate': (51, 100, 'Yellow', 'Unusually sensitive people should reduce outdoor activity'),
    'Unhealthy for Sensitive Groups': (101, 150, 'Orange', 'Active children and adults should limit outdoor activity'),
    'Unhealthy': (151, 200, 'Red', 'Everyone should limit outdoor activity'),
    'Very Unhealthy': (201, 300, 'Purple', 'Health alert: everyone may experience serious health effects'),
    'Hazardous': (301, 500, 'Maroon', 'Health warning: emergency conditions')
}

def generate_realistic_data(start_date='2023-01-01', end_date='2024-12-31'):
    """Generate realistic air pollution and wind data for Surigao del Sur"""
    
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    dates = pd.date_range(start=start, end=end, freq='D')
    
    all_data = []
    
    for municipality, info in MUNICIPALITIES.items():
        print(f"Generating data for {municipality}...")
        
        for date in dates:
            # Seasonal patterns (dry season: Nov-Apr, wet season: May-Oct)
            month = date.month
            is_dry = month in [11, 12, 1, 2, 3, 4]
            is_mining = info['mining_active']
            
            # Wind patterns in Surigao del Sur
            # Prevailing winds: Northeast monsoon (Nov-Mar), Southwest monsoon (Jun-Oct)
            if month in [11, 12, 1, 2, 3]:
                wind_direction = random.choice(['NE', 'NNE', 'ENE'])
                wind_speed = random.uniform(5, 15)  # km/h
            elif month in [6, 7, 8, 9, 10]:
                wind_direction = random.choice(['SW', 'WSW', 'SSW'])
                wind_speed = random.uniform(8, 20)  # km/h during monsoon
            else:
                wind_direction = random.choice(['E', 'ESE', 'SE'])
                wind_speed = random.uniform(3, 10)
            
            # Base pollution levels
            if is_mining:
                # Higher pollution in mining areas
                base_pm25 = random.uniform(30, 80)
                base_pm10 = random.uniform(50, 120)
                base_no2 = random.uniform(20, 60)
                base_so2 = random.uniform(15, 45)
            else:
                # Lower pollution in non-mining areas
                base_pm25 = random.uniform(10, 40)
                base_pm10 = random.uniform(20, 70)
                base_no2 = random.uniform(10, 30)
                base_so2 = random.uniform(5, 25)
            
            # Wind effect: if wind blows from mining area, increase pollution
            # Simplified: higher wind speed spreads dust more
            wind_effect = 1 + (wind_speed / 100)  # 1-1.2x multiplier
            
            # Weather effects
            if is_dry:
                # Dry season = more dust
                weather_multiplier = random.uniform(1.0, 1.3)
            else:
                # Wet season = less dust (rain washes particles)
                weather_multiplier = random.uniform(0.7, 1.0)
            
            # Day of week effect (weekdays have more mining activity)
            is_weekday = date.weekday() < 5
            weekday_multiplier = 1.2 if is_weekday and is_mining else 1.0
            
            # Calculate final pollution values
            pm25 = base_pm25 * wind_effect * weather_multiplier * weekday_multiplier
            pm10 = base_pm10 * wind_effect * weather_multiplier * weekday_multiplier
            no2 = base_no2 * wind_effect * weather_multiplier * weekday_multiplier
            so2 = base_so2 * wind_effect * weather_multiplier * weekday_multiplier
            
            # Calculate AQI
            aqi = calculate_aqi(pm25)
            
            # Determine AQI category
            aqi_category = get_aqi_category(aqi)
            
            row = {
                'date': date.strftime('%Y-%m-%d'),
                'municipality': municipality,
                'mining_active': is_mining,
                'pm25': round(pm25, 1),
                'pm10': round(pm10, 1),
                'no2': round(no2, 1),
                'so2': round(so2, 1),
                'aqi': round(aqi),
                'aqi_category': aqi_category,
                'wind_speed_kmh': round(wind_speed, 1),
                'wind_direction': wind_direction,
                'temperature_c': round(random.uniform(24, 33), 1),
                'humidity_pct': round(random.uniform(65, 95), 1)
            }
            all_data.append(row)
    
    df = pd.DataFrame(all_data)
    return df

def calculate_aqi(pm25):
    """Calculate AQI from PM2.5 concentration (simplified)"""
    if pm25 <= 12:
        return pm25 * (50/12)
    elif pm25 <= 35.4:
        return 50 + (pm25 - 12) * (100-50)/(35.4-12)
    elif pm25 <= 55.4:
        return 100 + (pm25 - 35.4) * (150-100)/(55.4-35.4)
    elif pm25 <= 150.4:
        return 150 + (pm25 - 55.4) * (200-150)/(150.4-55.4)
    elif pm25 <= 250.4:
        return 200 + (pm25 - 150.4) * (300-200)/(250.4-150.4)
    else:
        return 300 + (pm25 - 250.4) * (500-300)/(350.4-250.4)

def get_aqi_category(aqi):
    """Get AQI category based on value"""
    if aqi <= 50:
        return 'Good'
    elif aqi <= 100:
        return 'Moderate'
    elif aqi <= 150:
        return 'Unhealthy for Sensitive Groups'
    elif aqi <= 200:
        return 'Unhealthy'
    elif aqi <= 300:
        return 'Very Unhealthy'
    else:
        return 'Hazardous'

def save_data():
    """Generate and save data"""
    os.makedirs('data', exist_ok=True)
    
    print("="*60)
    print("🌍 AIR POLLUTION DATA COLLECTION")
    print("For Surigao del Sur")
    print("="*60)
    
    df = generate_realistic_data()
    
    # Save to CSV
    df.to_csv('data/pollution_data.csv', index=False)
    
    print(f"\n✅ Data saved to data/pollution_data.csv")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Summary statistics
    print("\n📊 Summary Statistics:")
    print(f"   Average PM2.5: {df['pm25'].mean():.1f} µg/m³")
    print(f"   Average PM10: {df['pm10'].mean():.1f} µg/m³")
    print(f"   Average Wind Speed: {df['wind_speed_kmh'].mean():.1f} km/h")
    
    print("\n🏭 Mining vs Non-Mining Areas:")
    mining_avg = df[df['mining_active']]['pm25'].mean()
    non_mining_avg = df[~df['mining_active']]['pm25'].mean()
    print(f"   Mining areas PM2.5: {mining_avg:.1f} µg/m³")
    print(f"   Non-mining areas PM2.5: {non_mining_avg:.1f} µg/m³")
    
    return df

if __name__ == "__main__":
    df = save_data()
    print("\n🚀 Next: Run python train_model.py")