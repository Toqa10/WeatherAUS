# streamlit_app.py
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
    page_title="Weather AUS Prediction",
    layout="wide",
    page_icon="🌦️"
)

# ----------------------------
# CSS for light blue gradient + clouds & rain overlay
# ----------------------------
st.markdown("""
    <style>
    body {
        background: linear-gradient(to bottom, #f0fbff, #ffffff);
        color: #0c1e3d;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #0c1e3d;
        color: white;
    }
    .overlay {
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        pointer-events: none;
        z-index: -1;
        background: url('https://i.ibb.co/WD7zC6X/clouds.png') repeat-x,
                    url('https://i.ibb.co/W6DhwkQ/rain.png') repeat-y;
        background-size: contain, cover;
        opacity: 0.3;
    }
    </style>
    <div class="overlay"></div>
""", unsafe_allow_html=True)

# ----------------------------
# Load Data (Random Example for Multiple Cities)
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
st.title("🌦️ Weather AUS Prediction Dashboard")
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
    fig1 = px.histogram(filtered_df, x="Temp3pm", nbins=20, title="Temperature at 3 PM",
                        color_discrete_sequence=["#a0d8ff"])
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("### 💧 Rainfall Distribution")
    fig2 = px.histogram(filtered_df, x="Rainfall", nbins=20, title="Rainfall (mm)",
                        color_discrete_sequence=["#66b3ff"])
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 🌬️ Humidity Distribution")
    fig3 = px.histogram(filtered_df, x="Humidity3pm", nbins=20, title="Humidity at 3 PM (%)",
                        color_discrete_sequence=["#99ccff"])
    st.plotly_chart(fig3, use_container_width=True)

    # ----------------------------
    # Show Data
    # ----------------------------
    st.markdown("### 🗂️ Data Preview")
    st.dataframe(filtered_df[['Location','Month','Temp3pm','Rainfall','Humidity3pm','Predicted_RainTomorrow']].head(10))
else:
    st.warning("لا توجد بيانات لهذه المدينة والشهر المحددين.")
