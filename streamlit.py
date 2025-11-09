import requests
import numpy as np
import pandas as pd
import streamlit as st
from datetime import date, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import LSTM, Dense, Input #type: ignore

WINDOW_SIZE = 7
EPOCHS = 15
BATCH_SIZE = 16

DEFAULT_LAT = 30.0444
DEFAULT_LON = 31.2357
DEFAULT_TIMEZONE = None
DEFAULT_CITY = "Default Cairo"


def suggest_outfit(temp, rain):
    if rain is None:
        rain = 0.0
    if rain > 2.0:
        return "الجو ممطر… خُد جاكيت وشمسية"
    if temp is None:
        return "ما فيش بيانات كفاية"
    if temp < 10:
        return "برد قوي… جاكيت تقيل وبلوفر"
    if temp < 18:
        return "بارد… خفيف مع جاكيت"
    if temp < 26:
        return "لطيف… تيشيرت وجينز"
    if temp < 32:
        return "دافي… تيشيرت خفيف"
    return "حر جدًا… شورت وتيشيرت خفيف"


@st.cache_data(ttl=3600)
def get_location_by_ip():
    apis = [
        {"url": "https://ipapi.co/json/", "name": "ipapi"},
        {"url": "http://ip-api.com/json/", "name": "ip_api"},
    ]

    for api in apis:
        try:
            res = requests.get(api["url"], timeout=6)
            res.raise_for_status()
            data = res.json()

            # ----- ipapi -----
            if api["name"] == "ipapi":
                lat = data.get("latitude")
                lon = data.get("longitude")
                timezone = data.get("timezone")
                city = data.get("city")

            # ----- ip-api -----
            elif api["name"] == "ip_api":
                lat = data.get("lat")
                lon = data.get("lon")
                timezone = data.get("timezone")
                city = data.get("city")

            if lat is not None and lon is not None:
                return {
                    "lat": float(lat),
                    "lon": float(lon),
                    "timezone": timezone,
                    "city": city or "Unknown"
                }

        except:
            continue

    return None

@st.cache_data(ttl=3600 * 6)
def fetch_archive(lat, lon, start, end, timezone):
    daily_vars = "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max"

    url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start}&end_date={end}"
        f"&daily={daily_vars}"
        f"{'' if not timezone else f'&timezone={timezone}'}"
    )

    try:
        res = requests.get(url, timeout=15)
        res.raise_for_status()
        return res.json()
    except:
        return None


st.set_page_config(page_title="Weather Predictor", layout="centered")
st.title("🌤️ تطبيق التنبؤ بالطقس")


loc = get_location_by_ip()

if isinstance(loc, dict) and loc.get("lat") is not None and loc.get("lon") is not None:
    lat = loc["lat"]
    lon = loc["lon"]
    timezone = loc.get("timezone") or DEFAULT_TIMEZONE
    city = loc.get("city") or DEFAULT_CITY
else:
    lat = DEFAULT_LAT
    lon = DEFAULT_LON
    timezone = DEFAULT_TIMEZONE
    city = DEFAULT_CITY

st.write(f"📍 **الموقع الحالي:** {city}")

days_ahead = st.number_input("عدد الأيام للتنبؤ:", min_value=1, max_value=30, value=1)
start_btn = st.button("ابدأ التنبؤ")

if start_btn:
    with st.spinner("جاري جلب البيانات وتدريب النموذج…"):

        start = "2020-01-01"
        end = str(date.today())

        raw = fetch_archive(lat, lon, start, end, timezone)

        if not raw or "daily" not in raw:
            st.error("تعذر جلب البيانات من API.")
            st.stop()

        df = pd.DataFrame(raw["daily"])

        required = {
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "windspeed_10m_max",
            "time"
        }

        if not required.issubset(df.columns):
            st.error("البيانات ناقصة من API.")
            st.stop()

        df["temp_mean"] = (df["temperature_2m_max"] + df["temperature_2m_min"]) / 2
        df = df[["time", "temp_mean", "precipitation_sum", "windspeed_10m_max"]]

        features = df[["temp_mean", "precipitation_sum", "windspeed_10m_max"]].astype(float).values

        if len(features) <= WINDOW_SIZE:
            st.error("البيانات قليلة جدًا للتدريب.")
            st.stop()

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(features)

        X, y = [], []
        for i in range(len(scaled) - WINDOW_SIZE):
            X.append(scaled[i:i + WINDOW_SIZE])
            y.append(scaled[i + WINDOW_SIZE, 0])

        X = np.array(X)
        y = np.array(y)

        model = Sequential([
            Input(shape=(WINDOW_SIZE, X.shape[2])),
            LSTM(64, return_sequences=True),
            LSTM(32),
            Dense(1)
        ])

        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

        last = scaled[-WINDOW_SIZE:].copy()
        predictions = []

        for _ in range(days_ahead):
            pred = model.predict(np.expand_dims(last, axis=0), verbose=0)
            inv = np.zeros((1, 3))
            inv[0, 0] = pred[0][0]
            temp_real = scaler.inverse_transform(inv)[0, 0]

            predictions.append(temp_real)

            new_row = np.array([[pred[0][0], last[-1][1], last[-1][2]]])
            last = np.vstack((last[1:], new_row))

        rain_last = float(df.iloc[-1]["precipitation_sum"])
        wind_last = float(df.iloc[-1]["windspeed_10m_max"])

        st.success("✅ التنبؤ جاهز")
        results = []

        for i, temp in enumerate(predictions, start=1):
            date_future = (date.today() + timedelta(days=i)).strftime("%d-%m-%Y")
            outfit = suggest_outfit(temp, rain_last)
            results.append({
                "التاريخ": date_future,
                "درجة الحرارة": f"{temp:.1f}°C",
                "الاقتراح": outfit
            })

        st.table(pd.DataFrame(results))
