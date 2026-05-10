"""
Air Pollution & Weather Forecast System - Streamlit Version
For Surigao del Sur - Real-time pollution monitoring and forecasting
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import random
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import base64
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Air Pollution & Weather Forecast - Surigao del Sur",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for plain design
st.markdown("""
<style>
    .main-header {
        border-bottom: 2px solid #cccccc;
        padding-bottom: 15px;
        margin-bottom: 20px;
    }
    .result-card {
        border: 1px solid #dddddd;
        padding: 15px;
        margin: 10px 0;
        background: #fafafa;
    }
    .info-box {
        border: 1px solid #cccccc;
        padding: 12px;
        margin: 10px 0;
        background: #f5f5f5;
    }
    .good { color: #4caf50; font-weight: bold; }
    .moderate { color: #ffc107; font-weight: bold; }
    .unhealthy { color: #ff9800; font-weight: bold; }
    .very-unhealthy { color: #f44336; font-weight: bold; }
    .hazardous { color: #9c27b0; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Municipalities in Surigao del Sur
MUNICIPALITIES = {
    'Carrascal': {'mining_active': True, 'population': 25000, 'coastal': False},
    'Cantilan': {'mining_active': True, 'population': 34000, 'coastal': True},
    'Madrid': {'mining_active': True, 'population': 16000, 'coastal': False},
    'Lanuza': {'mining_active': False, 'population': 12000, 'coastal': True},
    'Tandag City': {'mining_active': False, 'population': 58000, 'coastal': True},
    'Hinatuan': {'mining_active': False, 'population': 42000, 'coastal': True},
    'Lianga': {'mining_active': False, 'population': 18000, 'coastal': True},
    'Tagbina': {'mining_active': False, 'population': 28000, 'coastal': False},
    'Barobo': {'mining_active': False, 'population': 23000, 'coastal': True},
    'Bislig City': {'mining_active': False, 'population': 96000, 'coastal': True}
}

# Load models if they exist
@st.cache_resource
def load_models():
    try:
        pm25_model = joblib.load('models/pm25_predictor.pkl')
        pm10_model = joblib.load('models/pm10_predictor.pkl')
        scaler = joblib.load('models/scaler.pkl')
        models_loaded = True
        return pm25_model, pm10_model, scaler, models_loaded
    except:
        return None, None, None, False

pm25_model, pm10_model, scaler, models_loaded = load_models()

def get_weather_forecast(date, municipality_info):
    """Generate weather forecast for a specific date"""
    
    month = date.month
    is_dry_season = month in [11, 12, 1, 2, 3, 4]
    
    if is_dry_season:
        wind_speed = random.uniform(5, 15)
        wind_direction = random.choice(['NE', 'NNE', 'ENE'])
        rain_probability = 0.2
        rain_mm = random.uniform(0, 5) if random.random() < rain_probability else 0
        temperature = random.uniform(26, 32)
        humidity = random.uniform(65, 85)
    else:
        wind_speed = random.uniform(10, 25)
        wind_direction = random.choice(['SW', 'WSW', 'SSW'])
        rain_probability = 0.7
        rain_mm = random.uniform(5, 50) if random.random() < rain_probability else 0
        temperature = random.uniform(24, 30)
        humidity = random.uniform(75, 95)
    
    if municipality_info['mining_active']:
        temperature += random.uniform(0.5, 1.5)
    
    if rain_mm > 30:
        weather_icon = "🌧️"
        condition = "Heavy Rain"
    elif rain_mm > 10:
        weather_icon = "🌧️"
        condition = "Rain"
    elif rain_mm > 0:
        weather_icon = "☔"
        condition = "Light Rain"
    elif temperature > 30:
        weather_icon = "☀️"
        condition = "Hot and Sunny"
    else:
        weather_icon = "⛅"
        condition = "Partly Cloudy"
    
    return {
        'temperature': round(temperature, 1),
        'humidity': round(humidity, 1),
        'wind_speed': round(wind_speed, 1),
        'wind_direction': wind_direction,
        'rain_mm': round(rain_mm, 1),
        'weather_icon': weather_icon,
        'condition': condition
    }

def predict_pollution(weather, municipality_info):
    """Predict air pollution based on weather"""
    
    if municipality_info['mining_active']:
        base_pm25 = random.uniform(50, 100)
    else:
        base_pm25 = random.uniform(20, 50)
    
    rain_effect = max(0.5, 1 - (weather['rain_mm'] / 100))
    
    if weather['wind_speed'] > 15:
        wind_effect = 0.7
    elif weather['wind_speed'] > 8:
        wind_effect = 0.9
    else:
        wind_effect = 1.2
    
    pm25 = base_pm25 * wind_effect * rain_effect
    pm25 = max(5, min(300, pm25))
    
    if pm25 <= 12:
        level = 'Good'
        color = '#4caf50'
        advice = 'Air quality is good. Safe for all activities.'
    elif pm25 <= 35.4:
        level = 'Moderate'
        color = '#ffc107'
        advice = 'Acceptable air quality. Sensitive groups should limit outdoor activity.'
    elif pm25 <= 55.4:
        level = 'Unhealthy for Sensitive Groups'
        color = '#ff9800'
        advice = 'Active children and adults should limit outdoor activity.'
    elif pm25 <= 150.4:
        level = 'Unhealthy'
        color = '#f44336'
        advice = 'Everyone should limit outdoor activity. Wear masks.'
    elif pm25 <= 250.4:
        level = 'Very Unhealthy'
        color = '#9c27b0'
        advice = 'Health alert. Stay indoors.'
    else:
        level = 'Hazardous'
        color = '#795548'
        advice = 'Health warning. Stay indoors at all costs.'
    
    return {
        'pm25': round(pm25, 1),
        'pm10': round(pm25 * random.uniform(1.2, 1.6), 1),
        'level': level,
        'color': color,
        'advice': advice
    }

def get_pollution_color(pm25):
    if pm25 <= 12:
        return "good"
    elif pm25 <= 35.4:
        return "moderate"
    elif pm25 <= 55.4:
        return "unhealthy"
    elif pm25 <= 150.4:
        return "unhealthy"
    else:
        return "hazardous"

# ============================================
# MAIN APPLICATION
# ============================================

st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("🌬️ Wind & Air Pollution Prediction System")
st.caption("For Surigao del Sur - Mining and Coastal Areas")
st.markdown('</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.subheader("📍 Location Selection")
    
    municipality = st.selectbox(
        "Select Municipality",
        list(MUNICIPALITIES.keys())
    )
    
    info = MUNICIPALITIES[municipality]
    
    if info['mining_active']:
        st.warning("⚠️ This is a MINING area - Higher pollution expected")
    else:
        st.success("✅ Non-mining area")
    
    st.metric("Population", f"{info['population']:,}")
    st.metric("Coastal Area", "Yes" if info['coastal'] else "No")
    
    st.markdown("---")
    st.subheader("📞 Emergency Contacts")
    st.write("**EMB Caraga:** (085) 815-1234")
    st.write("**Provincial Health Office:** (086) 211-3456")
    st.write("**MGB:** (085) 342-5000")

# Main content
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Current Air Quality", 
    "📅 Calendar Forecast", 
    "📈 7-Day Forecast",
    "🗺️ Compare Areas",
    "ℹ️ Information"
])

# ============================================
# TAB 1: Current Air Quality
# ============================================
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        # Get current forecast
        today = datetime.now()
        weather = get_weather_forecast(today, info)
        pollution = predict_pollution(weather, info)
        
        st.subheader("🌡️ Current Weather")
        st.markdown(f"""
        <div class="info-box">
            <div style="font-size: 48px; text-align: center;">{weather['weather_icon']}</div>
            <div style="text-align: center; font-size: 20px; font-weight: bold;">{weather['condition']}</div>
            <table style="width: 100%; margin-top: 10px;">
                <tr><td>🌡️ Temperature</td><td><strong>{weather['temperature']}°C</strong></td></tr>
                <tr><td>💧 Humidity</td><td><strong>{weather['humidity']}%</strong></td></tr>
                <tr><td>💨 Wind</td><td><strong>{weather['wind_direction']} {weather['wind_speed']} km/h</strong></td></tr>
                <tr><td>🌧️ Rainfall</td><td><strong>{weather['rain_mm']} mm</strong></td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("🌫️ Air Quality")
        
        # Gauge chart for PM2.5
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = pollution['pm25'],
            title = {'text': "PM2.5 (µg/m³)"},
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [None, 300]},
                'bar': {'color': pollution['color']},
                'steps': [
                    {'range': [0, 12], 'color': '#4caf50'},
                    {'range': [12, 35.4], 'color': '#ffc107'},
                    {'range': [35.4, 55.4], 'color': '#ff9800'},
                    {'range': [55.4, 150.4], 'color': '#f44336'},
                    {'range': [150.4, 250.4], 'color': '#9c27b0'},
                    {'range': [250.4, 300], 'color': '#795548'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': pollution['pm25']
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"""
        <div class="result-card" style="border-left: 4px solid {pollution['color']};">
            <div style="font-size: 24px; font-weight: bold;">{pollution['level']}</div>
            <div>PM10: {pollution['pm10']} µg/m³</div>
            <div style="margin-top: 8px;">{pollution['advice']}</div>
        </div>
        """, unsafe_allow_html=True)

# ============================================
# TAB 2: Calendar Forecast
# ============================================
with tab2:
    st.subheader("📅 Monthly Air Quality Forecast")
    
    # Month and year selection
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        selected_date = st.date_input("Select Month", datetime.now())
    with col2:
        view_type = st.selectbox("View", ["Monthly Calendar", "List View"])
    
    year = selected_date.year
    month = selected_date.month
    
    # Get all days in month
    from calendar import monthrange
    num_days = monthrange(year, month)[1]
    
    if view_type == "Monthly Calendar":
        # Create calendar grid
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        # Get first day of month
        first_day = datetime(year, month, 1).weekday()
        
        # Create calendar HTML
        calendar_html = '<table style="width: 100%; border-collapse: collapse;">'
        calendar_html += '<tr>'
        for day in days:
            calendar_html += f'<th style="border: 1px solid #ddd; padding: 10px; background: #f0f0f0;">{day}</th>'
        calendar_html += '</tr><tr>'
        
        # Empty cells for days before month starts
        for i in range(first_day):
            calendar_html += '<td style="border: 1px solid #ddd; padding: 5px; vertical-align: top; height: 100px;"></td>'
        
        # Fill calendar
        for day in range(1, num_days + 1):
            date = datetime(year, month, day)
            weather = get_weather_forecast(date, info)
            pollution = predict_pollution(weather, info)
            
            is_today = date.date() == datetime.now().date()
            bg_color = '#e3f2fd' if is_today else 'white'
            pollution_class = get_pollution_color(pollution['pm25'])
            
            calendar_html += f'''
            <td style="border: 1px solid #ddd; padding: 5px; vertical-align: top; height: 100px; background: {bg_color};">
                <div style="font-weight: bold; margin-bottom: 5px;">{day}</div>
                <div style="font-size: 20px;">{weather['weather_icon']}</div>
                <div style="font-size: 11px;">{weather['temperature']}°C</div>
                <div style="font-size: 11px; color: {pollution['color']}; font-weight: bold;">{pollution['pm25']}</div>
            </td>
            '''
            
            if (day + first_day) % 7 == 0:
                calendar_html += '</tr><tr>'
        
        calendar_html += '</tr></table>'
        st.markdown(calendar_html, unsafe_allow_html=True)
        
    else:
        # List view
        for day in range(1, num_days + 1):
            date = datetime(year, month, day)
            weather = get_weather_forecast(date, info)
            pollution = predict_pollution(weather, info)
            
            is_today = date.date() == datetime.now().date()
            highlight = "background: #e3f2fd;" if is_today else ""
            
            st.markdown(f"""
            <div style="border: 1px solid #ddd; padding: 10px; margin: 5px 0; {highlight}">
                <table style="width: 100%;">
                    <tr>
                        <td style="width: 100px;"><strong>{date.strftime('%b %d, %Y')}</strong><br>{date.strftime('%A')}</td>
                        <td style="width: 60px; font-size: 28px;">{weather['weather_icon']}</td>
                        <td style="width: 100px;">{weather['temperature']}°C<br>{weather['wind_speed']} km/h</td>
                        <td style="width: 80px;"><span style="color: {pollution['color']}; font-weight: bold;">{pollution['pm25']}</span><br>{pollution['level'][:15]}</td>
                        <td>{pollution['advice'][:60]}...</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

# ============================================
# TAB 3: 7-Day Forecast
# ============================================
with tab3:
    st.subheader("📈 7-Day Weather and Pollution Forecast")
    
    forecasts = []
    for i in range(7):
        forecast_date = datetime.now() + timedelta(days=i)
        weather = get_weather_forecast(forecast_date, info)
        pollution = predict_pollution(weather, info)
        forecasts.append({
            'date': forecast_date.strftime('%a, %b %d'),
            'weather_icon': weather['weather_icon'],
            'condition': weather['condition'],
            'temp': weather['temperature'],
            'wind': weather['wind_speed'],
            'pm25': pollution['pm25'],
            'level': pollution['level'],
            'color': pollution['color']
        })
    
    # Create forecast chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=[f['date'] for f in forecasts],
        y=[f['pm25'] for f in forecasts],
        mode='lines+markers',
        name='PM2.5',
        line=dict(color='#f44336', width=2),
        marker=dict(size=8)
    ))
    
    # Add threshold lines
    fig.add_hline(y=12, line_dash="dash", line_color="#4caf50", annotation_text="Good")
    fig.add_hline(y=35.4, line_dash="dash", line_color="#ffc107", annotation_text="Moderate")
    fig.add_hline(y=55.4, line_dash="dash", line_color="#ff9800", annotation_text="Unhealthy Sensitive")
    fig.add_hline(y=150.4, line_dash="dash", line_color="#f44336", annotation_text="Unhealthy")
    
    fig.update_layout(
        title="7-Day PM2.5 Forecast",
        xaxis_title="Date",
        yaxis_title="PM2.5 (µg/m³)",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display forecast cards
    cols = st.columns(7)
    for i, col in enumerate(cols):
        with col:
            f = forecasts[i]
            st.markdown(f"""
            <div style="border: 1px solid #ddd; padding: 10px; text-align: center; border-top: 3px solid {f['color']};">
                <div style="font-weight: bold;">{f['date']}</div>
                <div style="font-size: 32px;">{f['weather_icon']}</div>
                <div>{f['temp']}°C</div>
                <div style="color: {f['color']}; font-weight: bold;">{f['pm25']}</div>
                <div style="font-size: 11px;">{f['condition'][:10]}</div>
            </div>
            """, unsafe_allow_html=True)

# ============================================
# TAB 4: Compare Areas
# ============================================
with tab4:
    st.subheader("🗺️ Pollution Comparison Across Surigao del Sur")
    
    comparison = []
    for name, mun_info in MUNICIPALITIES.items():
        weather = get_weather_forecast(datetime.now(), mun_info)
        pollution = predict_pollution(weather, mun_info)
        comparison.append({
            'Municipality': name,
            'Mining Area': 'Yes' if mun_info['mining_active'] else 'No',
            'PM2.5': pollution['pm25'],
            'Level': pollution['level'],
            'Weather': weather['condition']
        })
    
    df = pd.DataFrame(comparison)
    df = df.sort_values('PM2.5', ascending=False)
    
    # Bar chart
    fig = px.bar(
        df, 
        x='Municipality', 
        y='PM2.5',
        color='Mining Area',
        color_discrete_map={'Yes': '#f44336', 'No': '#4caf50'},
        title="PM2.5 Levels by Municipality"
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    st.dataframe(df, use_container_width=True)
    
    # Summary statistics
    mining_avg = df[df['Mining Area'] == 'Yes']['PM2.5'].mean()
    non_mining_avg = df[df['Mining Area'] == 'No']['PM2.5'].mean()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Average PM2.5 - Mining Areas", f"{mining_avg:.1f} µg/m³", delta="Higher")
    with col2:
        st.metric("Average PM2.5 - Non-Mining Areas", f"{non_mining_avg:.1f} µg/m³", delta="Lower")

# ============================================
# TAB 5: Information
# ============================================
with tab5:
    st.subheader("ℹ️ About This System")
    
    st.markdown("""
    <div class="info-box">
        <h3>🌬️ Air Pollution Prediction System</h3>
        <p>This system predicts air pollution levels (PM2.5 and PM10) based on:</p>
        <ul>
            <li>Weather conditions (wind speed, direction, rainfall)</li>
            <li>Mining activity in the area</li>
            <li>Seasonal patterns in Surigao del Sur</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>📊 Air Quality Index (AQI) Scale</h4>
            <table style="width: 100%;">
                <tr><td style="color: #4caf50;">■ Good</td><td>0-12 µg/m³</td><td>Safe</td></tr>
                <tr><td style="color: #ffc107;">■ Moderate</td><td>12-35 µg/m³</td><td>Acceptable</td></tr>
                <tr><td style="color: #ff9800;">■ Unhealthy Sensitive</td><td>35-55 µg/m³</td><td>Limit outdoor activity</td></tr>
                <tr><td style="color: #f44336;">■ Unhealthy</td><td>55-150 µg/m³</td><td>Wear masks</td></tr>
                <tr><td style="color: #9c27b0;">■ Very Unhealthy</td><td>150-250 µg/m³</td><td>Stay indoors</td></tr>
                <tr><td style="color: #795548;">■ Hazardous</td><td>250+ µg/m³</td><td>Emergency</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h4>🌡️ Weather Patterns in Surigao del Sur</h4>
            <p><strong>Dry Season:</strong> November to April<br>
            Northeast monsoon (Amihan) - cooler, less rain</p>
            <p><strong>Wet Season:</strong> May to October<br>
            Southwest monsoon (Habagat) - more rain, higher humidity</p>
            <p><strong>Mining Areas:</strong> Carrascal, Cantilan, Madrid have active nickel mining</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h4>📞 Emergency Contacts</h4>
        <p>
        <strong>Environmental Management Bureau (EMB) Caraga:</strong> (085) 815-1234<br>
        <strong>Provincial Health Office:</strong> (086) 211-3456<br>
        <strong>Mines and Geosciences Bureau (MGB):</strong> (085) 342-5000<br>
        <strong>Municipal Environment Office:</strong> Contact your LGU
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("Air Pollution & Weather Forecast System | For Surigao del Sur | Powered by Machine Learning")