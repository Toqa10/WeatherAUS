# weather_dashboard.py
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
    page_title="Weather AUS Prediction",
    layout="wide",
    page_icon="🌦️"
)

# ----------------------------
# CSS للثيم والخلفية المتحركة
# ----------------------------
st.markdown("""
<style>
/* اجعل body يغطي الصفحة كلها */
body, .stApp {
    height: 100%;
    margin: 0;
    background: #cceeff; /* لبني فاتح */
    overflow: hidden;
}

/* Canvas يغطي كامل الصفحة */
#bgCanvas {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: -1;
}

/* Text Styles */
h1, h2, h3, h4, h5, h6, p, label {
    color: #034f84;
    font-weight: bold;
}
</style>

<canvas id="bgCanvas"></canvas>

<script>
// إعداد canvas
const canvas = document.getElementById('bgCanvas');
const ctx = canvas.getContext('2d');
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

// ------------------ المطر ------------------
const drops = [];
for (let i = 0; i < 200; i++) {
    drops.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        length: 10 + Math.random() * 20,
        speed: 4 + Math.random() * 4,
        opacity: 0.2 + Math.random() * 0.3
    });
}

// ------------------ السحب ------------------
const clouds = [];
for (let i = 0; i < 10; i++) {
    clouds.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height * 0.3,
        radius: 30 + Math.random() * 50,
        speed: 0.2 + Math.random() * 0.3
    });
}

// ------------------ الرسم ------------------
function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // رسم السحب
    clouds.forEach(c => {
        ctx.beginPath();
        ctx.fillStyle = "rgba(255,255,255,0.6)";
        ctx.arc(c.x, c.y, c.radius, 0, Math.PI * 2);
        ctx.fill();
        c.x += c.speed;
        if(c.x - c.radius > canvas.width) c.x = -c.radius;
    });
    
    // رسم المطر
    drops.forEach(d => {
        ctx.beginPath();
        ctx.strokeStyle = `rgba(173,216,230,${d.opacity})`;
        ctx.lineWidth = 2;
        ctx.moveTo(d.x, d.y);
        ctx.lineTo(d.x, d.y + d.length);
        ctx.stroke();
        d.y += d.speed;
        if(d.y > canvas.height) d.y = -d.length;
    });

    requestAnimationFrame(animate);
}

animate();

// ------------------ تعديل حجم الصفحة ------------------
window.addEventListener('resize', () => {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
});
</script>
""", unsafe_allow_html=True)

# ----------------------------
# بيانات عشوائية لتجربة الداشبورد
# ----------------------------
@st.cache_data
def load_data():
    data = {
        "Location": np.random.choice(["Sydney","Melbourne","Brisbane","Perth","Adelaide"], 300),
        "Month": np.random.randint(1,13,300),
        "RainTomorrow": np.random.choice(["Yes","No"], 300),
        "Rainfall": np.random.rand(300)*20,
        "Temp3pm": np.random.rand(300)*15+15,
        "Humidity3pm": np.random.randint(30,100,300)
    }
    df = pd.DataFrame(data)
    return df

df = load_data()

# Encode
le_location = LabelEncoder()
df['Location_enc'] = le_location.fit_transform(df['Location'])
le_target = LabelEncoder()
df['RainTomorrow_enc'] = le_target.fit_transform(df['RainTomorrow'])

# Features & Model
features = ['Location_enc','Month','Rainfall','Temp3pm','Humidity3pm']
X = df[features]
y = df['RainTomorrow_enc']
model = LogisticRegression()
model.fit(X, y)

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("Filters")
selected_location = st.sidebar.selectbox("Select Location", df['Location'].unique())
selected_month = st.sidebar.selectbox("Select Month", sorted(df['Month'].unique()))
filtered_df = df[(df['Location']==selected_location) & (df['Month']==selected_month)]

# ----------------------------
# Title
# ----------------------------
st.title("🌦️ Weather AUS Prediction Dashboard")
st.subheader(f"Location: {selected_location} | Month: {selected_month}")

# Prediction
X_pred = filtered_df[features]
y_pred = model.predict(X_pred)
filtered_df['Predicted_RainTomorrow'] = le_target.inverse_transform(y_pred)

rain_count = filtered_df['Predicted_RainTomorrow'].value_counts()
pred_text = "🌧️ Rain Tomorrow" if rain_count.get("Yes",0) > rain_count.get("No",0) else "☀️ No Rain"
st.markdown(f"<h2 style='color:#034f84'>{pred_text}</h2>", unsafe_allow_html=True)

# Charts
st.markdown("### 🌡️ Temperature Distribution")
fig1 = px.histogram(filtered_df, x="Temp3pm", nbins=20, title="Temperature at 3 PM", color_discrete_sequence=["#66c2ff"])
st.plotly_chart(fig1, use_container_width=True)

st.markdown("### 💧 Rainfall Distribution")
fig2 = px.histogram(filtered_df, x="Rainfall", nbins=20, title="Rainfall (mm)", color_discrete_sequence=["#3399ff"])
st.plotly_chart(fig2, use_container_width=True)

st.markdown("### 🌬️ Humidity Distribution")
fig3 = px.histogram(filtered_df, x="Humidity3pm", nbins=20, title="Humidity at 3 PM (%)", color_discrete_sequence=["#99ccff"])
st.plotly_chart(fig3, use_container_width=True)

# Show Data
st.markdown("### 🗂️ Data Preview")
st.dataframe(filtered_df[['Location','Month','Temp3pm','Rainfall','Humidity3pm','Predicted_RainTomorrow']].head(10))
import streamlit as st
import pickle
import numpy as np

# ---------- Page Config ----------
st.set_page_config(page_title="WeatherAUS", page_icon="🌧️", layout="centered")

# ---------- Load Models ----------
def load_models():
    try:
        model = pickle.load(open("Decision Tree.pkl", "rb"))
        rain_today_encoder = pickle.load(open("RainToday_label_encoder.pkl", "rb"))
        rain_tomorrow_encoder = pickle.load(open("RainTomorrow_label_encoder.pkl", "rb"))
        return model, rain_today_encoder, rain_tomorrow_encoder
    except Exception as e:
        st.error("❌ Error loading models/encoders. Make sure pickle files exist.")
        return None, None, None

model, rain_today_encoder, rain_tomorrow_encoder = load_models()

# ---------- Navigation ----------
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_home():
    st.session_state.page = "home"

def go_prediction():
    st.session_state.page = "prediction"

# ---------- Home Page ----------
if st.session_state.page == "home":
    st.title("🌤️ WeatherAUS Dashboard")
    st.write("Welcome to the **WeatherAUS App**!")
    st.write("From here you can go to the prediction page.")

    st.button("🔮 Go to Prediction", on_click=go_prediction)

# ---------- Prediction Page ----------
elif st.session_state.page == "prediction":
    st.title("🌧️ RainTomorrow Prediction")
    st.write("Enter today's weather data to predict if it will rain tomorrow.")

    MaxTemp = st.number_input("Max Temperature (°C)", value=25.0, step=0.1)
    Rainfall = st.number_input("Rainfall (mm)", value=0.0, step=0.1)
    WindGustSpeed = st.number_input("Wind Gust Speed (km/h)", value=35.0, step=1.0)
    Humidity9am = st.number_input("Humidity at 9AM (%)", value=60.0, step=1.0)
    Humidity3pm = st.number_input("Humidity at 3PM (%)", value=55.0, step=1.0)
    Pressure9am = st.number_input("Pressure at 9AM (hPa)", value=1015.0, step=0.1)
    Pressure3pm = st.number_input("Pressure at 3PM (hPa)", value=1013.0, step=0.1)
    Temp3pm = st.number_input("Temperature at 3PM (°C)", value=22.0, step=0.1)
    RainToday = st.selectbox("Rain Today?", ["No", "Yes"])
    RISK_MM = st.number_input("RISK_MM (mm)", value=0.2, step=0.1)

    if st.button("Predict"):
        if model and rain_today_encoder and rain_tomorrow_encoder:
            RainToday_encoded = rain_today_encoder.transform([RainToday])[0]
            features = np.array([
                MaxTemp, Rainfall, WindGustSpeed, Humidity9am, Humidity3pm,
                Pressure9am, Pressure3pm, Temp3pm, RainToday_encoded, RISK_MM
            ]).reshape(1, -1)

            prediction = model.predict(features)[0]
            prediction_label = rain_tomorrow_encoder.inverse_transform([prediction])[0]
            prob = model.predict_proba(features)[0]

            st.success(f"☁️ Prediction: **{prediction_label}**")
            st.info(f"📊 Probability → No: {prob[0]*100:.2f}% | Yes: {prob[1]*100:.2f}%")
        else:
            st.error("⚠️ Model or encoder not loaded.")

    st.button("🏠 Back to Home", on_click=go_home)

