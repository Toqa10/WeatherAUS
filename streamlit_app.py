# streamlit_app_animated.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Weather AUS Animated",
    layout="wide",
    page_icon="🌦️"
)

# ----------------------------
# CSS for animated clouds & rain
# ----------------------------
st.markdown("""
    <style>
    body {
        background: linear-gradient(to bottom, #f0fbff, #ffffff);
        color: #0c1e3d;
        font-family: 'Arial', sans-serif;
        overflow-x: hidden;
    }

    .stButton>button {
        background-color: #0c1e3d;
        color: white;
    }

    /* Rain animation */
    @keyframes rainFall {
        0% {top: -10%; }
        100% {top: 100%; }
    }

    .raindrop {
        position: absolute;
        width: 2px;
        height: 15px;
        background: #66b3ff;
        animation: rainFall linear infinite;
    }

    /* Clouds animation */
    @keyframes cloudMove {
        0% {left: -20%; }
        100% {left: 100%; }
    }

    .cloud {
        position: absolute;
        top: 10%;
        width: 200px;
        height: 60px;
        background: url('https://i.ibb.co/WD7zC6X/clouds.png') no-repeat;
        background-size: cover;
        animation: cloudMove 60s linear infinite;
    }

    </style>

    <div class="cloud" style="top:5%; animation-duration: 90s;"></div>
    <div class="cloud" style="top:20%; animation-duration: 120s;"></div>
    <div class="cloud" style="top:35%; animation-duration: 100s;"></div>

    <!-- Generate 50 raindrops -->
    """ + "\n".join([f'<div class="raindrop" style="left:{i*2}%; animation-duration:{np.random.randint(1,3)}s;"></div>' for i in range(50)]) + """
""", unsafe_allow_html=True)

# ----------------------------
# Load Data (Random Example)
# ----------------------------
@st.cache_data
def load_data():
    cities = ["Sydney","Melbourne","Brisbane","Perth","Adelaide","Hobart","Darwin","Cairns"]
    data = {
        "Location": np.random.choice(cities, 500),
        "Month": np.random.randint(1,13,500),
        "RainTomorrow": np.random.choice(["Yes","No"], 500),
        "Rainfall": np.random.rand(500)*25,
        "Temp3pm": np.random.rand(500)*15+15,
        "Humidity3pm": np.random.randint(30,100,500)
    }
    df = pd.DataFrame(data)
    return df

df = load_data()

# ----------------------------
# Encode categorical
# ----------------------------
le_location = LabelEncoder()
df['Location_enc'] = le_location.fit_transform(df['Location'])

le_target = LabelEncoder()
df['RainTomorrow_enc'] = le_target.fit_transform(df['RainTomorrow'])

# ----------------------------
# Features & Model
# ----------------------------
features = ['Location_enc','Month','Rainfall','Temp3pm','Humidity3pm']
X = df[features]
y = df['RainTomorrow_enc']

model = LogisticRegression()
model.fit(X, y)

# ----------------------------
# Sidebar Filters
# ----------------------------
st.sidebar.header("Filters")
selected_location = st.sidebar.selectbox("Select Location", sorted(df['Location'].unique()))
selected_month = st.sidebar.selectbox("Select Month", sorted(df['Month'].unique()))

filtered_df = df[(df['Location']==selected_location) & (df['Month']==selected_month)]

# ----------------------------
# Title
# ----------------------------
st.title("🌦️ Weather AUS Animated Dashboard")
st.subheader(f"Location: {selected_location} | Month: {selected_month}")

# ----------------------------
# Prediction
# ----------------------------
if not filtered_df.empty:
    X_pred = filtered_df[features]
    y_pred = model.predict(X_pred)
    filtered_df = filtered_df.copy()
    filtered_df['Predicted_RainTomorrow'] = le_target.inverse_transform(y_pred)

    rain_count = filtered_df['Predicted_RainTomorrow'].value_counts()
    pred_text = "🌧️ Rain Tomorrow" if rain_count.get("Yes",0) > rain_count.get("No",0) else "☀️ No Rain"
    st.markdown(f"<h2 style='color:#0c1e3d'>{pred_text}</h2>", unsafe_allow_html=True)

    # ----------------------------
    # Charts
    # ----------------------------
    st.markdown("### 🌡️ Temperature Distribution")
    fig1 = px.histogram(filtered_df, x="Temp3pm", nbins=20, color_discrete_sequence=["#a0d8ff"])
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("### 💧 Rainfall Distribution")
    fig2 = px.histogram(filtered_df, x="Rainfall", nbins=20, color_discrete_sequence=["#66b3ff"])
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 🌬️ Humidity Distribution")
    fig3 = px.histogram(filtered_df, x="Humidity3pm", nbins=20, color_discrete_sequence=["#99ccff"])
    st.plotly_chart(fig3, use_container_width=True)

    # ----------------------------
    # Show Data
    # ----------------------------
    st.markdown("### 🗂️ Data Preview")
    st.dataframe(filtered_df[['Location','Month','Temp3pm','Rainfall','Humidity3pm','Predicted_RainTomorrow']].head(10))
else:
    st.warning("لا توجد بيانات لهذه المدينة والشهر المحددين.")
