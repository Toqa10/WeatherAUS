import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

# ---------- Streamlit Page Config ----------
st.set_page_config(page_title="WeatherAUS Rain Prediction",
                   layout="wide",
                   page_icon="🌧️")

st.title("🌦️ WeatherAUS Rain Prediction Dashboard")

# ---------- Load Data ----------
@st.cache_data
def load_data():
    df = pd.read_csv("weatherAUS.csv")
    # Encode categorical columns
    le_location = LabelEncoder()
    df['Location_enc'] = le_location.fit_transform(df['Location'])
    
    le_rain_today = LabelEncoder()
    df['RainToday_enc'] = le_rain_today.fit_transform(df['RainToday'])
    
    le_rain_tomorrow = LabelEncoder()
    df['RainTomorrow_enc'] = le_rain_tomorrow.fit_transform(df['RainTomorrow'])
    
    # Fill missing numeric values
    numeric_cols = ['Rainfall','WindGustSpeed','WindSpeed_mean','Humidity9am','Humidity3pm','Pressure3pm','Temp3pm','RISK_MM']
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # Fill missing categorical
    df['WindGustDir'] = df['WindGustDir'].fillna('None')
    df['WindDir9am'] = df['WindDir9am'].fillna('None')
    df['WindDir3pm'] = df['WindDir3pm'].fillna('None')
    
    return df, le_location, le_rain_today, le_rain_tomorrow

df, le_location, le_rain_today, le_rain_tomorrow = load_data()

# ---------- Background Gradient + Cloud/Mist Simulation ----------
fig_bg = go.Figure()
fig_bg.update_layout(
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    paper_bgcolor='lightblue',
    plot_bgcolor='lightblue',
    margin=dict(l=0,r=0,t=0,b=0),
    height=300
)
# Add simple cloud/mist effect using shapes
fig_bg.add_shape(type="circle", x0=0.1, y0=0.7, x1=0.3, y1=0.9, fillcolor="white", line_color="white")
fig_bg.add_shape(type="circle", x0=0.2, y0=0.6, x1=0.4, y1=0.8, fillcolor="white", line_color="white")
fig_bg.add_shape(type="circle", x0=0.6, y0=0.7, x1=0.8, y1=0.9, fillcolor="white", line_color="white")
fig_bg.add_shape(type="circle", x0=0.7, y0=0.6, x1=0.9, y1=0.8, fillcolor="white", line_color="white")
st.plotly_chart(fig_bg, use_container_width=True)

# ---------- Sidebar Filters ----------
st.sidebar.header("Filters")
selected_location = st.sidebar.selectbox("Select Location", df['Location'].unique())
selected_month = st.sidebar.selectbox("Select Month", sorted(df['Month'].dropna().unique()))

# ---------- Filter Data ----------
filtered_df = df[(df['Location']==selected_location) & (df['Month']==selected_month)]

# ---------- Train a simple model ----------
feature_cols = ['Rainfall','WindGustSpeed','WindSpeed_mean','Humidity9am','Humidity3pm','Pressure3pm','Temp3pm','RISK_MM','RainToday_enc']
X = df[feature_cols]
y = df['RainTomorrow_enc']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Predict on filtered data
filtered_df['Predicted_RainTomorrow'] = model.predict(filtered_df[feature_cols])
filtered_df['Predicted_RainTomorrow_label'] = le_rain_tomorrow.inverse_transform(filtered_df['Predicted_RainTomorrow'])

# ---------- Display Data ----------
st.subheader(f"Data Preview for {selected_location} in Month {selected_month}")
st.dataframe(filtered_df[['Location','Month','RainTomorrow','Predicted_RainTomorrow_label']].head(10))

# ---------- Charts ----------
st.subheader("RainTomorrow Count by Season")
season_cols = ['Season_Spring','Season_Summer','Season_Winter']
season_mapping = {'Season_Spring':'Spring','Season_Summer':'Summer','Season_Winter':'Winter'}
season_counts = {}
for col in season_cols:
    season_counts[season_mapping[col]] = filtered_df[filtered_df[col]==1]['RainTomorrow'].value_counts()
season_df = pd.DataFrame(season_counts).T.fillna(0)
season_df = season_df.rename(columns={'No':'No Rain','Yes':'Rain'})
st.bar_chart(season_df)

st.subheader("Humidity Distribution")
fig_hum = plt.figure(figsize=(6,4))
sns.histplot(filtered_df['Humidity9am'], bins=20, kde=True, color='skyblue')
st.pyplot(fig_hum)

st.subheader("Temperature Distribution")
fig_temp = plt.figure(figsize=(6,4))
sns.histplot(filtered_df['Temp3pm'], bins=20, kde=True, color='orange')
st.pyplot(fig_temp)

st.subheader("Rainfall vs Wind Speed")
fig_rw = plt.figure(figsize=(6,4))
sns.scatterplot(data=filtered_df, x='WindSpeed_mean', y='Rainfall', hue='Predicted_RainTomorrow_label', palette=['green','blue'])
st.pyplot(fig_rw)
