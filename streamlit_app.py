# -----------------------------
# Imports
# -----------------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBRFClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.decomposition import PCA
import plotly.express as px

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(layout="wide", page_title="🌦️ WeatherAUS Dashboard")

# -----------------------------
# CSS for background and rain
# -----------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to bottom, #a0c4ff, #caf0f8);
    background-image: url('https://i.ibb.co/HCd0xVH/clouds.png');
    background-size: cover;
}
/* Rain effect */
@keyframes rain {0% {top: -10%; opacity: 0;} 50% {opacity:1;} 100% {top:100%; opacity:0;}}
.raindrop {position:absolute; width:2px; height:15px; background:white; opacity:0.5; animation:rain linear infinite; animation-duration:1s;}
</style>
<script>
const body = document.body;
for(let i=0;i<150;i++){
  const drop=document.createElement('div');
  drop.className='raindrop';
  drop.style.left=Math.random()*100+'vw';
  drop.style.animationDuration=(0.5+Math.random()*1.5)+'s';
  drop.style.top=Math.random()*100+'vh';
  body.appendChild(drop);
}
</script>
""", unsafe_allow_html=True)

# -----------------------------
# Load dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("weatherAUS.csv")
    # Encode categorical
    df['RainTomorrow_Code'] = df['RainTomorrow'].map({'Yes':1,'No':0})
    df['RainToday_Code'] = df['RainToday'].map({'Yes':1,'No':0})
    le_loc = LabelEncoder()
    df['Location_Code'] = le_loc.fit_transform(df['Location'])
    # Create WindSpeed_mean if not exists
    if 'WindSpeed_mean' not in df.columns:
        df['WindSpeed_mean'] = df[['WindGustSpeed']].mean(axis=1)
    # Fill missing numerical values with median
    num_cols = ['Rainfall','Temp3pm','Humidity3pm','WindSpeed_mean']
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    return df, le_loc

df, le_location = load_data()

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.title("Filters")
selected_location = st.sidebar.selectbox("Select Location", df['Location'].unique())
selected_month = st.sidebar.slider("Select Month (1-12)", 1, 12, 1)

filtered_df = df[(df['Location']==selected_location) & (df['Month']==selected_month)]

st.title(f"🌦️ Weather Dashboard: {selected_location} | Month {selected_month}")

# -----------------------------
# Rain Probability Metric
# -----------------------------
rain_prob = filtered_df['RainTomorrow_Code'].mean() if len(filtered_df)>0 else 0
st.metric("Probability of RainTomorrow", f"{rain_prob*100:.2f}%")

# -----------------------------
# Weather Summary
# -----------------------------
st.write("### Weather Summary")
stats_cols = ['Rainfall','Temp3pm','Humidity3pm','WindSpeed_mean']
if len(filtered_df) > 0:
    st.dataframe(filtered_df[stats_cols].describe().T)
else:
    st.write("No data for this selection")

# -----------------------------
# Visualizations
# -----------------------------
st.write("### Visualizations")

# 1️⃣ RainTomorrow Count
fig1, ax1 = plt.subplots(figsize=(5,4))
if len(filtered_df) > 0:
    sns.countplot(data=filtered_df, x='RainTomorrow', palette='Set2', ax=ax1)
    ax1.set_title("RainTomorrow Distribution")
st.pyplot(fig1)

# 2️⃣ Temp3pm Distribution (Violin)
fig2, ax2 = plt.subplots(figsize=(5,4))
if len(filtered_df) > 0:
    sns.violinplot(x='RainTomorrow', y='Temp3pm', data=filtered_df, palette='Set3', ax=ax2)
st.pyplot(fig2)

# 3️⃣ Humidity3pm Density (KDE)
fig3, ax3 = plt.subplots(figsize=(6,4))
if len(filtered_df) > 0:
    sns.kdeplot(filtered_df['Humidity3pm'].dropna(), fill=True, color='skyblue', ax=ax3)
st.pyplot(fig3)

# 4️⃣ Rainfall vs WindSpeed Scatter
fig4, ax4 = plt.subplots(figsize=(6,4))
if len(filtered_df) > 0:
    sns.scatterplot(data=filtered_df, x='WindSpeed_mean', y='Rainfall', hue='RainTomorrow', palette='coolwarm', ax=ax4)
st.pyplot(fig4)

# -----------------------------
# ML Models Training
# -----------------------------
st.write("### Model Training & Evaluation")
X = df[['Rainfall','Temp3pm','Humidity3pm','WindSpeed_mean']]
y = df['RainTomorrow_Code']
X_scaled = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "DecisionTree": DecisionTreeClassifier(),
    "SVC": SVC(probability=True),
    "KNN": KNeighborsClassifier(),
    "GaussianNB": GaussianNB(),
    "XGBRFClassifier": XGBRFClassifier(use_label_encoder=False, eval_metric='logloss'),
    "LGBMClassifier": LGBMClassifier(),
    "CatBoostClassifier": CatBoostClassifier(verbose=0),
    "ExtraTreesClassifier": ExtraTreesClassifier()
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred)
    })

res_df = pd.DataFrame(results)
st.dataframe(res_df)

# -----------------------------
# PCA + Clustering
# -----------------------------
st.write("### PCA & KMeans Clustering")
pca = PCA(n_components=2)
pca_res = pca.fit_transform(X_scaled)
kmeans = ExtraTreesClassifier(n_estimators=100, random_state=42).fit(X_scaled)  # example clustering placeholder
fig5 = px.scatter(x=pca_res[:,0], y=pca_res[:,1], color=y, hover_data=[df['Location']], title="PCA 2D Plot")
st.plotly_chart(fig5, use_container_width=True)

# -----------------------------
# Scenario Simulation
# -----------------------------
st.write("### Scenario Simulation")
temp_input = st.slider("Temp3pm (°C)", int(df['Temp3pm'].min()), int(df['Temp3pm'].max()), 25)
humidity_input = st.slider("Humidity3pm (%)", int(df['Humidity3pm'].min()), int(df['Humidity3pm'].max()), 70)
windspeed_input = st.slider("WindSpeed_mean (km/h)", int(df['WindSpeed_mean'].min()), int(df['WindSpeed_mean'].max()), 15)
rainfall_input = st.slider("Rainfall (mm)", 0, int(df['Rainfall'].max()), 0)

subset = df[
    (df['Temp3pm'].between(temp_input-2, temp_input+2)) &
    (df['Humidity3pm'].between(humidity_input-5, humidity_input+5)) &
    (df['WindSpeed_mean'].between(windspeed_input-5, windspeed_input+5)) &
    (df['Rainfall'].between(rainfall_input-2, rainfall_input+2))
]
sim_prob = subset['RainTomorrow_Code'].mean() if len(subset)>0 else 0
st.metric("Simulated Rain Probability", f"{sim_prob*100:.2f}%")

# -----------------------------
# Raw Data
# -----------------------------
with st.expander("Show Raw Data"):
    st.dataframe(filtered_df)
