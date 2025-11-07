# app.py
import streamlit as st
import pandas as pd
import joblib

# === Load model ===
model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.title("💼 Adult Income Prediction App")
st.write("Prediksi apakah seseorang berpenghasilan >50K atau <=50K berdasarkan data demografis.")

# === Input dari user ===
age = st.number_input("Umur", min_value=17, max_value=90, value=30)
education_num = st.slider("Level Pendidikan (Education Num)", 1, 16, 9)
hours_per_week = st.slider("Jam kerja per minggu", 1, 100, 40)
capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
capital_loss = st.number_input("Capital Loss", min_value=0, value=0)

# contoh fitur kategori:
workclass = st.selectbox("Workclass", ["Private", "Self-emp-not-inc", "Local-gov", "State-gov", "Federal-gov"])
occupation = st.selectbox("Occupation", ["Exec-managerial", "Prof-specialty", "Adm-clerical", "Sales", "Craft-repair"])
sex = st.selectbox("Jenis Kelamin", ["Male", "Female"])

# === Buat dataframe input ===
input_data = pd.DataFrame({
    "age": [age],
    "education_num": [education_num],
    "hours_per_week": [hours_per_week],
    "capital_gain": [capital_gain],
    "capital_loss": [capital_loss],
    "workclass": [workclass],
    "occupation": [occupation],
    "sex": [sex]
})

# Pastikan kolom sesuai dengan model (kalau perlu encode)
# Kalau dataset kamu sudah one-hot encoded, di sini harus ikut di-encode juga sesuai mapping model

st.write("### Data yang kamu input:")
st.dataframe(input_data)

# === Prediksi ===
if st.button("Prediksi Penghasilan"):
    pred = model.predict(input_data)[0]
    income_label = label_encoder.inverse_transform([pred])[0]
    st.success(f"💰 Prediksi: {income_label}")
