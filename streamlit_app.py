# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import os

# إعداد الصفحة
st.set_page_config(page_title="Weather AUS Prediction", layout="wide")

# 🎨 تصميم الواجهة
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to bottom, #a2d5f2, #ffffff);
        color: #000;
    }
    .stDataFrame div {
        color: black;
    }
    </style>
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

# تحويل الأعمدة النصية لأرقام
if 'RainTomorrow' in df.columns:
    le_rain = LabelEncoder()
    df['RainTomorrow_enc'] = le_rain.fit_transform(df['RainTomorrow'])
else:
    st.error("⚠️ Column 'RainTomorrow' not found!")
    st.stop()

# Sidebar: الفلاتر
st.sidebar.header("Filters")
locations = df['Location'].dropna().unique()
selected_location = st.sidebar.selectbox("Select Location", locations)

if 'Month' in df.columns and df['Month'].notna().any():
    months = sorted(df['Month'].dropna().unique())
    selected_month = st.sidebar.selectbox("Select Month", months)
else:
    months = []
    selected_month = None
    st.sidebar.info("Month filter ignored (column missing).")

# فلترة الداتا حسب البلد والشهر
if selected_month is not None:
    filtered_df = df[(df['Location']==selected_location) & (df['Month']==selected_month)]
else:
    filtered_df = df[df['Location']==selected_location]

st.markdown(f"### Weather data for {selected_location}" + (f", Month {selected_month}" if selected_month else ""))
st.dataframe(filtered_df)

# 👁️ Visualizations باستخدام filtered_df

# 1️⃣ RainTomorrow by Location
if 'RainTomorrow' in filtered_df.columns:
    fig1 = px.histogram(filtered_df, x='Location', color='RainTomorrow', barmode='group', title="RainTomorrow by Location")
    st.plotly_chart(fig1, use_container_width=True)

# 2️⃣ RainTomorrow by Season
season_cols = ['Season_Spring','Season_Summer','Season_Winter']
season_map = {'Season_Spring':'Spring','Season_Summer':'Summer','Season_Winter':'Winter'}
season_counts = {}
for col in season_cols:
    if col in filtered_df.columns:
        season_counts[season_map[col]] = filtered_df[filtered_df[col]==1]['RainTomorrow'].value_counts()
if season_counts:
    season_df = pd.DataFrame(season_counts).T.fillna(0)
    st.bar_chart(season_df)

# 3️⃣ Rainfall distribution
if 'Rainfall' in filtered_df.columns:
    fig3 = px.histogram(filtered_df, x='Rainfall', nbins=50, title="Rainfall Distribution")
    st.plotly_chart(fig3, use_container_width=True)

# 4️⃣ WindSpeed_mean distribution
if 'WindSpeed_mean' in filtered_df.columns:
    fig4 = px.histogram(filtered_df, x='WindSpeed_mean', nbins=30, title="WindSpeed Mean Distribution")
    st.plotly_chart(fig4, use_container_width=True)

# 5️⃣ Temp3pm distribution
if 'Temp3pm' in filtered_df.columns:
    fig5 = px.histogram(filtered_df, x='Temp3pm', nbins=30, title="Temp3pm Distribution")
    st.plotly_chart(fig5, use_container_width=True)

# 🔮 Prediction Example
st.markdown("### Rain Prediction Example")
if st.button("Predict if it will rain tomorrow"):
    st.info("Using RandomForestClassifier (example)")
    features = ['Rainfall','WindGustSpeed','Humidity9am','Humidity3pm','Pressure3pm','Temp3pm','WindSpeed_mean']
    features = [f for f in features if f in filtered_df.columns]
    X = filtered_df[features].fillna(0)
    y = filtered_df['RainTomorrow_enc']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    pred = model.predict(X_scaled)
    filtered_df['Predicted_RainTomorrow'] = le_rain.inverse_transform(pred)

    st.success("✅ Prediction added!")

    # التحقق من الأعمدة قبل عرضها
    display_cols = ['Location','Predicted_RainTomorrow']
    if 'Month' in filtered_df.columns:
        display_cols.insert(1, 'Month')  # ضيف Month لو موجود

    st.dataframe(filtered_df[display_cols].head(10))
