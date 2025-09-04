import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import plotly.express as px

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="🌦️ Weather AUS Simple Rain",
    layout="wide",
    page_icon="🌦️"
)

# ----------------------------
# CSS: Light blue background + clouds + simple rain
# ----------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(to bottom, #cceeff, #e6f7ff);
    color: #0c1e3d;
    font-family: 'Arial', sans-serif;
    overflow-x: hidden;
}

/* Clouds */
.cloud {
    position: absolute;
    top: 10%;
    width: 200px;
    height: 80px;
    background: url('https://i.ibb.co/WD7zC6X/clouds.png') no-repeat;
    background-size: cover;
    opacity: 0.4;
    animation: cloudMove 120s linear infinite;
}

.cloud2 { top: 30%; width:250px; animation-duration: 150s; }
.cloud3 { top: 50%; width:180px; animation-duration: 100s; }

/* Clouds Animation */
@keyframes cloudMove {
    0% {left: -25%;}
    100% {left: 105%;}
}

/* Simple raindrops */
@keyframes rainFall {
    0% {top: -10%;}
    100% {top: 110%;}
}
.raindrop {
    position: absolute;
    width: 2px;
    height: 15px;
    background: #99d6ff;
    animation: rainFall linear infinite;
    opacity: 0.6;
    border-radius: 50%;
}
</style>

<div class="cloud"></div>
<div class="cloud cloud2"></div>
<div class="cloud cloud3"></div>

""" + "\n".join([
    f'<div class="raindrop" style="left:{i*3}%; animation-duration:{np.random.uniform(1.0,1.5)}s;"></div>'
    for i in range(30)
]) , unsafe_allow_html=True)

# ----------------------------
# Example Random Data
# ----------------------------
@st.cache_data
def load_data():
    data = {
        "Location": np.random.choice(
            ["Sydney","Melbourne","Brisbane","Perth","Adelaide","Hobart","Darwin","Canberra"], 400
        ),
        "Month": np.random.randint(1,13,400),
        "RainTomorrow": np.random.choice(["Yes","No"], 400),
        "Rainfall": np.random.rand(400)*20,
        "Temp3pm": np.random.rand(400)*15+15,
        "Humidity3pm": np.random.randint(30,100,400)
    }
    return pd.DataFrame(data)

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
selected_location = st.sidebar.selectbox("Select Location", df['Location'].unique())
selected_month = st.sidebar.selectbox("Select Month", sorted(df['Month'].unique()))

filtered_df = df[(df['Location']==selected_location) & (df['Month']==selected_month)]

# ----------------------------
# Title & Prediction
# ----------------------------
st.title("🌦️ Weather AUS Simple Rain Dashboard")
st.subheader(f"Location: {selected_location} | Month: {selected_month}")

X_pred = filtered_df[features]
y_pred = model.predict(X_pred)
filtered_df['Predicted_RainTomorrow'] = le_target.inverse_transform(y_pred)

rain_count = filtered_df['Predicted_RainTomorrow'].value_counts()
pred_text = "🌧️ Rain Tomorrow" if rain_count.get("Yes",0) > rain_count.get("No",0) else "☀️ No Rain"
st.markdown(f"<h2 style='color:#0c1e3d'>{pred_text}</h2>", unsafe_allow_html=True)

# ----------------------------
# Charts
# ----------------------------
st.markdown("### 🌡️ Temperature Distribution")
fig1 = px.histogram(filtered_df, x="Temp3pm", nbins=20, title="Temperature at 3 PM",
                    color_discrete_sequence=["#66c2ff"])
st.plotly_chart(fig1, use_container_width=True)

st.markdown("### 💧 Rainfall Distribution")
fig2 = px.histogram(filtered_df, x="Rainfall", nbins=20, title="Rainfall (mm)",
                    color_discrete_sequence=["#3399ff"])
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
