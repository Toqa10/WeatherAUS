# weather_chatbot_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import pickle

# ---- الصفحة الرئيسية ----
st.set_page_config(
    page_title="Weather Chatbot 🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS للتصميم: خلفية لبني فاتح مع تأثير المطر
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to bottom, #cceeff, #ffffff);
    }
    .css-1aumxhk {
        color: #034f84;
    }
    </style>
    """, unsafe_allow_html=True
)

st.title("Weather Chatbot 🌦️")
st.write("اسأل عن توقعات المطر لأي مدينة وأي شهر!")

# ---- تحميل البيانات ----
@st.cache_data
def load_data():
    # هنا هنستخدم نسخة صغيرة من البيانات لو موجودة
    df = pd.read_csv("weatherAUS_small.csv")  # نسخة صغيرة لتجنب مشاكل الحجم
    df['Month'] = df['Month'].astype(int)
    le_location = LabelEncoder()
    df['Location_enc'] = le_location.fit_transform(df['Location'])
    return df, le_location

df, le_location = load_data()

# ---- تحميل الموديل ----
@st.cache_resource
def load_model():
    with open("rain_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# ---- واجهة المستخدم ----
locations = df['Location'].unique()
months = sorted(df['Month'].unique())

selected_location = st.selectbox("اختر المدينة:", locations)
selected_month = st.selectbox("اختر الشهر (1-12):", months)

# فلترة الداتا حسب اختيار المستخدم
filtered_df = df[(df['Location']==selected_location) & (df['Month']==selected_month)]

# ---- التنبؤ ----
if not filtered_df.empty:
    X = filtered_df[['Rainfall','WindGustSpeed','Humidity9am','Humidity3pm','Pressure3pm','Temp3pm','WindSpeed_mean']]
    prediction = model.predict(X)
    filtered_df['Predicted_RainTomorrow'] = prediction

    st.subheader(f"توقع المطر في {selected_location} لشهر {selected_month}")
    rain_yes = np.sum(prediction=='Yes')
    rain_no = np.sum(prediction=='No')
    st.write(f"عدد الأيام المتوقع فيها مطر: {rain_yes}")
    st.write(f"عدد الأيام المتوقع فيها لا مطر: {rain_no}")

    # ---- Visualization ----
    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(x=prediction, palette=['#87ceeb','#034f84'], ax=ax)
    ax.set_title(f"توزيع الأيام حسب توقع المطر في {selected_location}")
    ax.set_xlabel("توقع المطر")
    ax.set_ylabel("عدد الأيام")
    st.pyplot(fig)
else:
    st.warning("لا توجد بيانات لهذه المدينة والشهر المحددين.")

st.info("🌦️ تقدر تغير المدينة أو الشهر لمعرفة التوقعات المختلفة!")
