# streamlit_rain_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import streamlit.components.v1 as components

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Weather AUS Prediction",
    layout="wide",
    page_icon="🌦️"
)

# ----------------------------
# Background Animation (Rain & Clouds)
# ----------------------------
rain_html = """
<canvas id="rainCanvas" style="position: fixed; top:0; left:0; width:100%; height:100%; z-index:-1;"></canvas>
<script>
const canvas = document.getElementById('rainCanvas');
const ctx = canvas.getContext('2d');
function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
}
resizeCanvas();
window.addEventListener('resize', resizeCanvas);

const drops = [];
for(let i=0;i<300;i++){
    drops.push({x: Math.random()*canvas.width, y: Math.random()*canvas.height, l: Math.random()*20+10, xs: Math.random()*2-1, ys: Math.random()*4+4});
}

function draw(){
    // Background light blue
    ctx.fillStyle = '#cceeff';
    ctx.fillRect(0,0,canvas.width,canvas.height);

    // Clouds
    ctx.fillStyle = 'rgba(255,255,255,0.7)';
    ctx.beginPath();
    ctx.arc(200,100,60,0,Math.PI*2);
    ctx.arc(260,100,60,0,Math.PI*2);
    ctx.arc(230,80,50,0,Math.PI*2);
    ctx.fill();

    ctx.beginPath();
    ctx.arc(600,150,50,0,Math.PI*2);
    ctx.arc(650,150,50,0,Math.PI*2);
    ctx.arc(625,130,40,0,Math.PI*2);
    ctx.fill();

    // Raindrops
    ctx.strokeStyle = 'rgba(153,214,255,0.8)';
    ctx.lineWidth = 2;
    for(let i=0;i<drops.length;i++){
        let d = drops[i];
        ctx.beginPath();
        ctx.moveTo(d.x,d.y);
        ctx.lineTo(d.x+d.xs, d.y+d.l);
        ctx.stroke();

        d.x += d.xs;
        d.y += d.ys;
        if(d.y > canvas.height){
            d.y = -20;
            d.x = Math.random()*canvas.width;
        }
    }
    requestAnimationFrame(draw);
}

draw();
</script>
"""
components.html(rain_html, height=600)

# ----------------------------
# Load Data (Random Example)
# ----------------------------
@st.cache_data
def load_data():
    data = {
        "Location": np.random.choice(["Sydney","Melbourne","Brisbane","Perth","Adelaide","Hobart"], 500),
        "Month": np.random.randint(1,13,500),
        "RainTomorrow": np.random.choice(["Yes","No"], 500),
        "Rainfall": np.random.rand(500)*20,
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
selected_location = st.sidebar.selectbox("Select Location", df['Location'].unique())
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
    filtered_df['Predicted_RainTomorrow'] = le_target.inverse_transform(y_pred)

    rain_count = filtered_df['Predicted_RainTomorrow'].value_counts()
    pred_text = "🌧️ Rain Tomorrow" if rain_count.get("Yes",0) > rain_count.get("No",0) else "☀️ No Rain"
    st.markdown(f"<h2 style='color:#034f84'>{pred_text}</h2>", unsafe_allow_html=True)

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
else:
    st.warning("No data available for the selected location and month.")
