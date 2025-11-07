import streamlit as st
import pandas as pd
import joblib

# === Load model & encoder ===
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

st.title("💼 Adult Income Prediction (Decision Tree)")

# === Input user ===
age = st.number_input("Umur", 17, 90, 30)
education_num = st.slider("Level Pendidikan (Education Num)", 1, 16, 9)
hours_per_week = st.slider("Jam kerja per minggu", 1, 100, 40)
capital_gain = st.number_input("Capital Gain", 0, value=0)
capital_loss = st.number_input("Capital Loss", 0, value=0)
workclass = st.selectbox("Workclass", ["Private", "Self-emp-not-inc", "Local-gov", "State-gov", "Federal-gov"])
occupation = st.selectbox("Occupation", ["Exec-managerial", "Prof-specialty", "Adm-clerical", "Sales", "Craft-repair"])
sex = st.selectbox("Jenis Kelamin", ["Male", "Female"])

input_data = pd.DataFrame({
    "age": [age],
    "education.num": [education_num],
    "hours.per.week": [hours_per_week],
    "capital.gain": [capital_gain],
    "capital.loss": [capital_loss],
    "workclass": [workclass],
    "occupation": [occupation],
    "sex": [sex]
})

st.write("### Data yang kamu input:")
st.dataframe(input_data)

# === Prediksi ===
if st.button("Prediksi Penghasilan", key="predict_DT"):
    try:
        for col in encoders.keys():
            if col in input_data.columns:
                input_data[col] = encoders[col].transform(input_data[col])

        pred = model.predict(input_data)[0]
        income_label = target_encoder.inverse_transform([pred])[0]
        st.success(f"💰 Prediksi: {income_label}")
    except Exception as e:
        st.error(f"⚠️ Terjadi error saat prediksi: {e}")

