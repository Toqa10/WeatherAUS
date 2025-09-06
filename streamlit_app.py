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

# Load the trained model
model = pickle.load(open("weather_model.pkl", "rb"))

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Weather Classification"])

# Home Page
if page == "Home":
    st.title("🌦️ Welcome to the Weather App")
    st.write("Use the sidebar to navigate to the Weather Classification page.")

# Weather Classification Page
elif page == "Weather Classification":
    st.title("🌧️ Weather Classification")
    st.write("Enter the weather details below and check the prediction.")

    # Example input fields (change according to your dataset features)
    temp = st.number_input("Temperature (°C)", -10, 50, 25)
    humidity = st.slider("Humidity (%)", 0, 100, 50)
    wind = st.slider("Wind Speed (km/h)", 0, 100, 10)

    if st.button("Predict"):
        # Convert inputs into array
        features = np.array([[temp, humidity, wind]])
        
        try:
            prediction = model.predict(features)[0]
            proba = model.predict_proba(features)[0]

            st.subheader("✅ Prediction Result")
            st.write(f"Prediction: **{prediction}**")
            
            # Show probabilities as a dictionary
            st.write("Class Probabilities:")
            probs_dict = {cls: f"{p*100:.2f}%" for cls, p in zip(model.classes_, proba)}
            st.json(probs_dict)

        except Exception as e:
            st.error(f"Error: {e}")

