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
# weather_predict_page_styled.py
import streamlit as st
import numpy as np
import pickle

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Weather AUS - RainTomorrow Prediction",
    layout="wide",
    page_icon="🌦️"
)

# ----------------------------
# CSS for background gradient + animated clouds & rain
# ----------------------------
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to bottom, #cceeff, #f7fcff);
        color: #034f84;
        font-family: 'Arial', sans-serif;
    }

    .stButton>button {
        background-color: #034f84;
        color: white;
        border-radius: 8px;
        height: 40px;
        width: 200px;
        font-size: 16px;
    }

    .overlay {
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        pointer-events: none;
        z-index: -1;
    }

    /* Clouds animation */
    .cloud {
        position: absolute;
        background: url('https://i.ibb.co/WD7zC6X/clouds.png') no-repeat;
        width: 200px;
        height: 100px;
        animation: moveClouds linear infinite;
    }

    @keyframes moveClouds {
        0% { left: -250px; }
        100% { left: 100%; }
    }

    /* Rain animation */
    .raindrop {
        position: absolute;
        width: 2px;
        height: 10px;
        background: #66ccff;
        animation: fall linear infinite;
        opacity: 0.5;
    }

    @keyframes fall {
        0% { top: -10px; }
        100% { top: 100%; }
    }
    </style>

    <div class="overlay">
        <div class="cloud" style="top:10%; animation-duration:90s;"></div>
        <div class="cloud" style="top:30%; animation-duration:120s;"></div>
        <div class="cloud" style="top:50%; animation-duration:100s;"></div>

        <!-- 50 raindrops -->
        """ + "\n".join([f'<div class="raindrop" style="left:{i*2}%; animation-duration:{0.5+np.random.rand()}s;"></div>' for i in range(50)]) + """
    </div>
""", unsafe_allow_html=True)

# ----------------------------
# Title
# ----------------------------
st.title("🌦️ Weather AUS - RainTomorrow Prediction")
st.write("Enter today's weather data to predict if it will rain tomorrow.")

# ----------------------------
# Load model & encoder
# ----------------------------
@st.cache_resource
def load_model():
    with open("rain_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_encoder():
    with open("rain_today_encoder.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
rain_today_encoder = load_encoder()

# ----------------------------
# User Inputs
# ----------------------------
MaxTemp = st.number_input("Max Temperature (°C)", value=25.0, step=0.1)
Rainfall = st.number_input("Rainfall (mm)", value=0.0, step=0.1)
WindGustSpeed = st.number_input("Wind Gust Speed (km/h)", value=38.8, step=1.0)
Humidity9am = st.number_input("Humidity at 9AM (%)", value=60.8, step=1.0)
Humidity3pm = st.number_input("Humidity at 3PM (%)", value=58.0, step=1.0)
Pressure9am = st.number_input("Pressure at 9AM (hPa)", value=1015.0, step=0.1)
Pressure3pm = st.number_input("Pressure at 3PM (hPa)", value=1013.0, step=0.1)
Temp3pm = st.number_input("Temperature at 3PM (°C)", value=22.0, step=0.1)
RainToday = st.selectbox("Rain Today?", ["No", "Yes"])
RISK_MM = st.number_input("RISK_MM (mm)", value=0.0, step=0.1)

# Encode RainToday
RainToday_encoded = rain_today_encoder.transform([RainToday])[0]

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict Rain Tomorrow"):
    features = np.array([MaxTemp, Rainfall, WindGustSpeed, Humidity9am, Humidity3pm,
                         Pressure9am, Pressure3pm, Temp3pm, RainToday_encoded, RISK_MM]).reshape(1, -1)
    prediction = model.predict(features)[0]

    if prediction == "Yes":
        st.success("🌧️ It is likely to rain tomorrow!")
    else:
        st.success("☀️ No rain expected tomorrow.")
