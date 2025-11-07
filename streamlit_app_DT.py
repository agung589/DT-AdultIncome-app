# app.py
import streamlit as st
import pandas as pd
import joblib

# === Load model ===
model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# === Input user ===
# (slider, selectbox, dsb...)

# === Buat dataframe input ===
input_data = pd.DataFrame({...})

st.write("### Data yang kamu input:")
st.dataframe(input_data)

# === Prediksi ===
if st.button("Prediksi Penghasilan"):
    try:
        # Encode kolom kategorikal sesuai encoder training
        for col in ['workclass', 'occupation', 'sex']:
            input_data[col] = label_encoder.transform(input_data[col])

        # Prediksi
        pred = model.predict(input_data)[0]
        income_label = label_encoder.inverse_transform([pred])[0]

        st.success(f"💰 Prediksi: {income_label}")

    except Exception as e:
        st.error(f"⚠️ Terjadi error saat prediksi: {e}")


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

st.write("### Data yang kamu input:")
st.dataframe(input_data)

# === Prediksi ===
if st.button("Prediksi Penghasilan"):
    try:
        for col in ['workclass', 'occupation', 'sex']:
            input_data[col] = label_encoder.transform(input_data[col])

        pred = model.predict(input_data)[0]
        income_label = label_encoder.inverse_transform([pred])[0]
        st.success(f"💰 Prediksi: {income_label}")

    except Exception as e:
        st.error(f"⚠️ Terjadi error saat prediksi: {e}")

