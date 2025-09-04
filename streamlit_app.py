# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import os

st.set_page_config(page_title="Weather AUS Prediction", layout="wide")

# 🎨 تصميم الواجهة مع المطر المتحرك
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to bottom, #a2d5f2, #ffffff);
    }
    /* المطر */
    .rain {
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      overflow: hidden;
      z-index: 9999;
    }
    .raindrop {
      position: absolute;
      width: 2px;
      height: 10px;
      background: white;
      opacity: 0.6;
      animation: fall linear infinite;
    }
    @keyframes fall {
      0% {transform: translateY(-10px);}
      100% {transform: translateY(100vh);}
    }
    </style>
    <div class="rain" id="rain"></div>
    <script>
    const rain = document.getElementById("rain");
    for(let i=0;i<150;i++){
        let drop = document.createElement("div");
        drop.className="raindrop";
        drop.style.left=Math.random()*100+"%";
        drop.style.animationDuration=(0.5+Math.random()*0.5)+"s";
        drop.style.animationDelay=(Math.random()*5)+"s";
        rain.appendChild(drop);
    }
    </script>
    """,
    unsafe_allow_html=True
)

st.title("🌧️ Weather Prediction Dashboard")
st.subheader("Predict if it will rain tomorrow in Australia")

# 📂 تحميل CSV
csv_path = "weatherAUS.csv"
if not os.path.exists(csv_path):
    st.error("⚠️ File weatherAUS.csv not found in app folder!")
    st.stop()

df = pd.read_csv(csv_path)

# تحويل أعمدة النصوص لأرقام
le_rain = LabelEncoder()
df['RainTomorrow_enc'] = le_rain.fit_transform(df['RainTomorrow'])

# Sidebar لاختيار Location و Month
st.sidebar.header("Filters")
locations = df['Location'].unique()
selected_location = st.sidebar.selectbox("Select Location", locations)
months = sorted(df['Month'].dropna().unique())
selected_month = st.sidebar.selectbox("Select Month", months)

# فلترة الداتا
filtered_df = df[(df['Location']==selected_location) & (df['Month']==selected_month)]

st.markdown(f"### Weather data for {selected_location}, Month {selected_month}")
st.dataframe(filtered_df)

# 👁️ Visualizations

# 1️⃣ RainTomorrow by Location
fig1 = px.histogram(df, x='Location', color='RainTomorrow', barmode='group')
st.plotly_chart(fig1, use_container_width=True)

# 2️⃣ RainTomorrow by Month
fig2 = px.histogram(df, x='Month', color='RainTomorrow', barmode='group')
st.plotly_chart(fig2, use_container_width=True)

# 3️⃣ RainTomorrow by Season
season_cols = ['Season_Spring','Season_Summer','Season_Winter']
season_map = {'Season_Spring':'Spring','Season_Summer':'Summer','Season_Winter':'Winter'}
season_counts = {}
for col in season_cols:
    if col in df.columns:
        season_counts[season_map[col]] = df[df[col]==1]['RainTomorrow'].value_counts()
season_df = pd.DataFrame(season_counts).T.fillna(0)
st.bar_chart(season_df)

# 4️⃣ Rainfall distribution
fig3 = px.histogram(df, x='Rainfall', nbins=50)
st.plotly_chart(fig3, use_container_width=True)

# 5️⃣ WindSpeed_mean distribution
fig4 = px.histogram(df, x='WindSpeed_mean', nbins=30)
st.plotly_chart(fig4, use_container_width=True)

# 6️⃣ Temp3pm distribution
fig5 = px.histogram(df, x='Temp3pm', nbins=30)
st.plotly_chart(fig5, use_container_width=True)

# 🔮 Prediction (مثال بسيط)
st.markdown("### Rain Prediction Example")
if st.button("Predict if it will rain tomorrow"):
    st.info("Using RandomForestClassifier (example)")
    # تجهيز الداتا
    features = ['Rainfall','WindGustSpeed','Humidity9am','Humidity3pm','Pressure3pm','Temp3pm','WindSpeed_mean']
    X = df[features].fillna(0)
    y = df['RainTomorrow_enc']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    pred = model.predict(X_scaled)
    df['Predicted_RainTomorrow'] = le_rain.inverse_transform(pred)
    st.success("Prediction added to dataframe!")
    st.dataframe(df[['Location','Month','Predicted_RainTomorrow']].head(10))
