# train_model.py
# -------------------------------
# Script untuk melatih model Decision Tree dari dataset Adult Income
# dan menyimpannya ke model.pkl

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# 1️⃣ Baca dataset
df = pd.read_csv('Dataset_adult_income_clean.csv')

# 2️⃣ Pisahkan fitur (X) dan target (y)
# Pastikan kolom target kamu bernama 'income' (<=50K / >50K)
X = df.drop('income', axis=1)
y = df['income']

# Ubah data kategorikal menjadi numerik (Label Encoding per kolom)
encoders = {}
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le  # simpan encoder per kolom

# Pastikan target juga numerik
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# Split dan training model
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(x_train, y_train)

# Simpan model dan semua encoder
joblib.dump(model, "model.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(target_encoder, "target_encoder.pkl")
print("✅ Model & Encoders berhasil disimpan!")
