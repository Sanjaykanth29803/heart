import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("heart.csv")

X = df.drop("target", axis=1)
y = df["target"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


model = RandomForestClassifier()
model.fit(X_scaled, y)


st.set_page_config(page_title="Heart Disease Prediction App", layout="wide")
st.markdown("<h1 style='text-align: center; color: white;'>❤️ Heart Disease Prediction & Visualizations </h1>", unsafe_allow_html=True)
st.markdown("---")

tab1, tab2 = st.tabs([" Prediction", " Data Visualizations"])

with tab1:
    st.subheader("Enter Patient Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 18, 100, 50)
        sex = st.radio("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
        cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
        trestbps = st.number_input("Resting Blood Pressure (trestbps)", 90, 200, 120)
        thalach = st.number_input("Max Heart Rate Achieved (thalach)", 70, 210, 150)

    with col2:
        chol = st.number_input("Serum Cholesterol (chol)", 100, 600, 240)
        fbs = st.radio("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
        restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2])
        exang = st.radio("Exercise Induced Angina (exang)", [0, 1])
        oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0, 0.1)

    with col3:
        slope = st.selectbox("Slope of ST Segment (slope)", [0, 1, 2])
        ca = st.selectbox("No. of Major Vessels (ca)", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia (thal)", [0, 1, 2])
        st.markdown("###")

    input_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg,
                           thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
    input_scaled = scaler.transform(input_data)

    
    result_message = " The patient is not likely to have heart disease."

    if st.button("Predict"):
        prediction = model.predict(input_scaled)
        if prediction[0] == 1:
            result_message = " The patient is likely to have heart disease."
        else:
            result_message = " The patient is not likely to have heart disease."

    
    st.info(result_message)

with tab2:
    st.subheader("Exploratory Data Visualizations")

    st.markdown("###  Heart Disease Count")
    fig1 = px.histogram(df, x="target", color="target", barmode="group",
                        labels={"target": "Heart Disease"}, title="Heart Disease (1) vs No Disease (0)")
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("###  Age vs. Disease")
    fig2 = px.box(df, x="target", y="age", color="target", 
                  labels={"target": "Heart Disease", "age": "Age"}, title="Age Distribution by Heart Disease Status")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("###  Chest Pain Type Distribution")
    fig3 = px.histogram(df, x="cp", color="target", barmode="group", 
                        labels={"cp": "Chest Pain Type"}, title="Chest Pain Type vs Heart Disease")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("###  Correlation Heatmap")
    fig4, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig4)

    st.markdown("###  Max Heart Rate vs Age")
    fig5 = px.scatter(df, x="age", y="thalach", color="target", 
                      title="Max Heart Rate vs Age by Heart Disease")
    st.plotly_chart(fig5, use_container_width=True)
