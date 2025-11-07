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

# 3️⃣ Ubah data kategorikal menjadi numerik (Label Encoding)
le = LabelEncoder()
for col in X.columns:
    if X[col].dtype == 'object':  # kalau kolomnya teks
        X[col] = le.fit_transform(X[col])

# Pastikan target juga numerik
if y.dtype == 'object':
    y = le.fit_transform(y)

# 4️⃣ Split data untuk training & testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5️⃣ Buat & latih model Decision Tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 6️⃣ Simpan model ke file pkl
joblib.dump(model, 'model.pkl')
print("✅ Model berhasil dilatih dan disimpan sebagai 'model.pkl'")

# 5️⃣ Buat & latih model Decision Tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 6️⃣ Simpan model & encoder ke file pkl
joblib.dump(model, 'model.pkl')
joblib.dump(le, 'label_encoder.pkl')   # tambahkan baris ini
print("✅ Model & LabelEncoder berhasil disimpan (model.pkl, label_encoder.pkl)")
