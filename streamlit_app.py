# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Weather AUS Prediction",
    layout="wide",
    page_icon="🌦️"
)

# ----------------------------
# CSS for bright gradient background, clouds & rain
# ----------------------------
st.markdown("""
    <style>
    body {
        background: linear-gradient(to bottom, #d0e7ff, #f0f9ff); /* أفتح من السابق */
        color: #0c1e3d;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #0c1e3d;
        color: white;
    }
    /* Clouds & Rain overlay */
    .overlay {
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        pointer-events: none;
        z-index: -1;
        background: url('https://i.ibb.co/WD7zC6X/clouds.png') repeat-x,
                    url('https://i.ibb.co/W6DhwkQ/rain.png') repeat-y;
        background-size: cover, cover;
        opacity: 0.15;
    }
    </style>
    <div class="overlay"></div>
""", unsafe_allow_html=True)

# ----------------------------
# Load Data (Example Data)
# ----------------------------
@st.cache_data
def load_data():
    # مثال بيانات لتجربة التطبيق
    data = {
        "Location": np.random.choice(["Sydney","Melbourne","Brisbane","Perth"], 200),
        "Month": np.random.randint(1,13,200),
        "RainTomorrow": np.random.choice(["Yes","No"], 200),
        "Rainfall": np.random.rand(200)*20,
        "Temp3pm": np.random.rand(200)*15+15,
        "Humidity3pm": np.random.randint(30,100,200)
    }
    df = pd.DataFrame(data)
    return df

df = load_data()

# ----------------------------
# Sidebar Filters
# ----------------------------
st.sidebar.header("Filters")
locations = df['Location'].unique()
selected_location = st.sidebar.selectbox("Select Location", locations)

months = sorted(df['Month'].unique())
selected_month = st.sidebar.selectbox("Select Month", months)

filtered_df = df[(df['Location']==selected_location) & (df['Month']==selected_month)]

# ----------------------------
# Main Title
# ----------------------------
st.title("🌦️ Weather AUS Prediction Dashboard")
st.subheader(f"Location: {selected_location} | Month: {selected_month}")

# ----------------------------
# Prediction Widget (Mock)
# ----------------------------
rain_count = filtered_df['RainTomorrow'].value_counts()
pred_text = "🌧️ Rain Tomorrow" if rain_count.get("Yes",0) > rain_count.get("No",0) else "☀️ No Rain"
st.markdown(f"<h2 style='color:#0c1e3d'>{pred_text}</h2>", unsafe_allow_html=True)

# ----------------------------
# Charts
# ----------------------------
st.markdown("### 🌡️ Temperature Distribution")
fig1 = px.histogram(filtered_df, x="Temp3pm", nbins=20, title="Temperature at 3 PM",
                    color_discrete_sequence=["#4da6ff"])
st.plotly_chart(fig1, use_container_width=True)

st.markdown("### 💧 Rainfall Distribution")
fig2 = px.histogram(filtered_df, x="Rainfall", nbins=20, title="Rainfall (mm)",
                    color_discrete_sequence=["#1f77b4"])
st.plotly_chart(fig2, use_container_width=True)

st.markdown("### 🌬️ Humidity Distribution")
fig3 = px.histogram(filtered_df, x="Humidity3pm", nbins=20, title="Humidity at 3 PM (%)",
                    color_discrete_sequence=["#7fbfff"])
st.plotly_chart(fig3, use_container_width=True)

# ----------------------------
# Show filtered data
# ----------------------------
st.markdown("### 🗂️ Data Preview")
st.dataframe(filtered_df.head(10))
