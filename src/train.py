import pandas as pd
import numpy as np
import re
import tldextract
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------
# 1. SAMPLE DATASET
# -------------------------------
data = {
    "url": [
        "http://example.com",
        "http://secure-login-paypal.com",
        "https://google.com",
        "http://verify-account-bank.xyz",
        "https://github.com",
        "http://login-facebook-security.com"
    ],
    "label": [0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

# -------------------------------
# 2. FEATURE EXTRACTION
# -------------------------------
def extract_features(url):
    features = []

    features.append(len(url))
    features.append(url.count('.'))
    features.append(url.count('-'))
    features.append(url.count('@'))
    features.append(url.count('?'))
    features.append(url.count('%'))

    features.append(1 if url.startswith('https') else 0)

    suspicious_words = ['login', 'verify', 'bank', 'secure', 'account']
    features.append(int(any(word in url.lower() for word in suspicious_words)))

    ext = tldextract.extract(url)
    features.append(len(ext.domain))
    features.append(len(ext.subdomain))

    return np.array(features)

# -------------------------------
# 3. TRAIN ML MODEL (Random Forest)
# -------------------------------
X_ml = np.array([extract_features(url) for url in df['url']])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X_ml, y, test_size=0.2)

ml_model = RandomForestClassifier(n_estimators=100)
ml_model.fit(X_train, y_train)

print("ML Model trained!")

# -------------------------------
# 4. TRAIN DL MODEL (LSTM)
# -------------------------------
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(df['url'])

X_dl = tokenizer.texts_to_sequences(df['url'])
X_dl = pad_sequences(X_dl, maxlen=100)

dl_model = Sequential([
    Embedding(input_dim=100, output_dim=32, input_length=100),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

dl_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
dl_model.fit(X_dl, y, epochs=5, verbose=1)

print("DL Model trained!")

# -------------------------------
# 5. HYBRID PREDICTION FUNCTION
# -------------------------------
def predict_url(url):
    # ML prediction
    ml_features = extract_features(url).reshape(1, -1)
    ml_pred = ml_model.predict_proba(ml_features)[0][1]

    # DL prediction
    seq = tokenizer.texts_to_sequences([url])
    seq = pad_sequences(seq, maxlen=100)
    dl_pred = dl_model.predict(seq)[0][0]

    # Combine
    final_score = (ml_pred + dl_pred) / 2

    return {
        "ML Score": float(ml_pred),
        "DL Score": float(dl_pred),
        "Final Score": float(final_score),
        "Result": "Phishing" if final_score > 0.5 else "Safe"
    }

# -------------------------------
# 6. USER INPUT
# -------------------------------
if __name__ == "__main__":
    url = input("Enter URL: ")
    result = predict_url(url)

    print("\n--- Result ---")
    for k, v in result.items():
        print(f"{k}: {v}")
