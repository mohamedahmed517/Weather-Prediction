import streamlit as st
import requests
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import date, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Input # type: ignore

WINDOW_SIZE = 14
EPOCHS = 10
BATCH_SIZE = 16

egypt_governorates = {
    "Cairo": (30.0444, 31.2357),
    "Alexandria": (31.2001, 29.9187),
    "Giza": (30.0131, 31.2089),
    "Tanta": (30.7885, 31.0019),
    "Menoufia": (30.5972, 30.9876),
    "Luxor": (25.6872, 32.6396),
    "Aswan": (24.0889, 32.8998),
    "Ismailia": (30.6043, 32.2723),
    "Port Said": (31.2653, 32.3019),
    "Suez": (29.9668, 32.5498),
    "Fayoum": (29.3084, 30.8428),
    "Mansoura": (31.0364, 31.3807),
    "Beni Suef": (29.0661, 31.0994),
    "Minya": (28.1099, 30.7503),
    "Assiut": (27.1801, 31.1837),
    "Sohag": (26.5560, 31.6948),
    "Qena": (26.1642, 32.7267),
    "Damietta": (31.4165, 31.8133),
    "Kafr El Sheikh": (31.1107, 30.9405),
    "Sharkia": (30.7325, 31.7194),
    "Beheira": (30.8480, 30.3430),
    "Dakahlia": (31.1656, 31.4913),
    "Gharbia": (30.8754, 31.0335),
    "North Sinai": (30.2825, 33.6176),
    "South Sinai": (28.5529, 33.9368),
    "Matrouh": (31.3543, 27.2373),
    "New Valley": (25.4449, 28.7460),
    "Red Sea": (26.9650, 33.8994)
}

def suggest_outfit(temp, rain, wind):
    if rain > 2.0:
        return "الجو ممطر ⛈️ - خُد شمسية وجاكيت خفيف"
    elif temp < 10:
        return "برد جدًا 🧥 - البس جاكيت تقيل وبلوفر"
    elif 10 <= temp < 18:
        return "جو بارد 🧣 - البس جاكيت أو بلوفر خفيف"
    elif 18 <= temp < 26:
        return "جو معتدل 🌤️ - تيشيرت وجينز كفاية"
    elif 26 <= temp < 32:
        return "جو دافي ☀️ - البس تيشيرت خفيف وبنطلون صيفي"
    else:
        return "حر جدًا 🥵 - البس شورت وتيشيرت خفيف واشرب مياه كتير"


st.title("🌦️ تطبيق التنبؤ بالطقس في محافظات مصر")
st.write("ادخل المحافظة وعدد الأيام اللي عايز تعرف التنبؤ ليها:")

city = st.selectbox("اختار المحافظة:", list(egypt_governorates.keys()))
days_ahead = st.number_input("عدد الأيام للتنبؤ:", min_value=1, value=1)
start_prediction = st.button("ابدأ التنبؤ")

# ---- Processing ----
if start_prediction:
    with st.spinner("⏳ جاري تحميل بيانات الطقس وتدريب النموذج..."):
        lat, lon = egypt_governorates[city]
        start = "2020-01-01"
        end = str(date.today())

        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}"
            f"&start_date={start}&end_date={end}"
            f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max"
            f"&timezone=Africa/Cairo"
        )

        res = requests.get(url)
        data = res.json()

        df = pd.DataFrame(data["daily"])
        df["temp_mean"] = (df["temperature_2m_max"] + df["temperature_2m_min"]) / 2
        df = df[["time", "temp_mean", "precipitation_sum", "windspeed_10m_max"]]

        features = df[["temp_mean", "precipitation_sum", "windspeed_10m_max"]].values
        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(features)

        X, y = [], []
        for i in range(len(features_scaled) - WINDOW_SIZE):
            X.append(features_scaled[i:i+WINDOW_SIZE])
            y.append(features_scaled[i+WINDOW_SIZE, 0])
        X, y = np.array(X), np.array(y)

        model = Sequential([
            Input(shape=(WINDOW_SIZE, X.shape[2])),
            LSTM(64, return_sequences=True),
            LSTM(32),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

        last_seq = features_scaled[-WINDOW_SIZE:]
        today = date.today()

        predictions = []
        last_seq_copy = last_seq.copy()

        for _ in range(days_ahead):
            pred_scaled = model.predict(np.expand_dims(last_seq_copy, axis=0), verbose=0)
            inv = np.zeros((1, features.shape[1]))
            inv[0, 0] = pred_scaled[0][0]
            predicted_temp = scaler.inverse_transform(inv)[0, 0]
            predictions.append(predicted_temp)
            new_row = np.array([[pred_scaled[0][0], last_seq_copy[-1][1], last_seq_copy[-1][2]]])
            last_seq_copy = np.vstack((last_seq_copy[1:], new_row))

        rain = df.iloc[-1]["precipitation_sum"]
        wind = df.iloc[-1]["windspeed_10m_max"]

    st.success(f"✅ تم الإنتهاء من التنبؤ لمدينة {city}")

    results = []
    for i, temp in enumerate(predictions, start=1):
        future_date = (today + timedelta(days=i)).strftime("%d-%m-%Y")
        outfit = suggest_outfit(temp, rain, wind)
        results.append({"التاريخ": future_date, "درجة الحرارة": f"{temp:.1f}°C", "الاقتراح": outfit})

    st.subheader("📅 توقعات الطقس:")
    st.table(pd.DataFrame(results))