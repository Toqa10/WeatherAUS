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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.decomposition import PCA
import plotly.express as px

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(layout="wide", page_title="🌦️ WeatherAUS Dashboard")

# -----------------------------
# Load dataset safely
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("weatherAUS.csv")
    # إزالة أي فراغات في أسماء الأعمدة
    df.columns = df.columns.str.strip()
    
    # تأكد وجود Month
    if 'Month' in df.columns:
        df['Month'] = df['Month'].astype(int)
    else:
        st.error("Column 'Month' not found in dataset!")

    # Encode categorical
    if 'RainTomorrow' in df.columns:
        df['RainTomorrow_Code'] = df['RainTomorrow'].map({'Yes':1,'No':0})
    if 'RainToday' in df.columns:
        df['RainToday_Code'] = df['RainToday'].map({'Yes':1,'No':0})
    if 'Location' in df.columns:
        le_loc = LabelEncoder()
        df['Location_Code'] = le_loc.fit_transform(df['Location'])
    else:
        le_loc = None

    # Fill missing numerical values
    num_cols = ['Rainfall','Temp3pm','Humidity3pm','WindSpeed_mean']
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    return df, le_loc

df, le_location = load_data()

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.title("Filters")
if 'Location' in df.columns:
    selected_location = st.sidebar.selectbox("Select Location", df['Location'].unique())
else:
    selected_location = None

if 'Month' in df.columns:
    selected_month = st.sidebar.slider("Select Month (1-12)", 1, 12, 1)
else:
    selected_month = None

# -----------------------------
# Filter data safely
# -----------------------------
if selected_location is not None and selected_month is not None:
    filtered_df = df[(df['Location']==selected_location) & (df['Month']==selected_month)]
else:
    filtered_df = df.copy()

# -----------------------------
# Dashboard Title
# -----------------------------
st.title(f"🌦️ Weather Dashboard: {selected_location} | Month {selected_month}")

# -----------------------------
# Rain Probability Metric
# -----------------------------
rain_prob = filtered_df['RainTomorrow_Code'].mean() if 'RainTomorrow_Code' in filtered_df.columns and len(filtered_df)>0 else 0
st.metric("Probability of RainTomorrow", f"{rain_prob*100:.2f}%")

# -----------------------------
# Weather Summary
# -----------------------------
st.write("### Weather Summary")
stats_cols = ['Rainfall','Temp3pm','Humidity3pm','WindSpeed_mean']
stats_cols = [col for col in stats_cols if col in filtered_df.columns]
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
if 'RainTomorrow' in filtered_df.columns and len(filtered_df) > 0:
    sns.countplot(data=filtered_df, x='RainTomorrow', palette='Set2', ax=ax1)
    ax1.set_title("RainTomorrow Distribution")
st.pyplot(fig1)

# 2️⃣ Temp3pm Distribution (Violin)
fig2, ax2 = plt.subplots(figsize=(5,4))
if 'Temp3pm' in filtered_df.columns and 'RainTomorrow' in filtered_df.columns and len(filtered_df) > 0:
    sns.violinplot(x='RainTomorrow', y='Temp3pm', data=filtered_df, palette='Set3', ax=ax2)
st.pyplot(fig2)

# 3️⃣ Humidity3pm Density (KDE)
fig3, ax3 = plt.subplots(figsize=(6,4))
if 'Humidity3pm' in filtered_df.columns and len(filtered_df) > 0:
    sns.kdeplot(filtered_df['Humidity3pm'].dropna(), fill=True, color='skyblue', ax=ax3)
st.pyplot(fig3)

# 4️⃣ Rainfall vs WindSpeed Scatter
fig4, ax4 = plt.subplots(figsize=(6,4))
if 'Rainfall' in filtered_df.columns and 'WindSpeed_mean' in filtered_df.columns and 'RainTomorrow' in filtered_df.columns and len(filtered_df) > 0:
    sns.scatterplot(data=filtered_df, x='WindSpeed_mean', y='Rainfall', hue='RainTomorrow', palette='coolwarm', ax=ax4)
st.pyplot(fig4)

# -----------------------------
# ML Models Training (simplified)
# -----------------------------
st.write("### Model Training & Evaluation")
feature_cols = ['Rainfall','Temp3pm','Humidity3pm','WindSpeed_mean']
feature_cols = [col for col in feature_cols if col in df.columns]
X = df[feature_cols]
y = df['RainTomorrow_Code']
X_scaled = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "DecisionTree": DecisionTreeClassifier(),
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
# PCA 2D Plot
# -----------------------------
st.write("### PCA 2D Plot")
pca = PCA(n_components=2)
pca_res = pca.fit_transform(X_scaled)
fig5 = px.scatter(x=pca_res[:,0], y=pca_res[:,1], color=y, hover_data=[df['Location']], title="PCA 2D Plot")
st.plotly_chart(fig5, use_container_width=True)

# -----------------------------
# Raw Data
# -----------------------------
with st.expander("Show Raw Data"):
    st.dataframe(filtered_df)
