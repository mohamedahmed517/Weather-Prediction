import streamlit as st
import streamlit.components.v1 as components
import requests
import pandas as pd
import numpy as np
from datetime import date, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Input # type: ignore

# -----------------------------
# إعداد الصفحة
# -----------------------------
st.set_page_config(page_title="Weather Predictor", layout="centered")
st.title("🌤️ تطبيق التنبؤ بالطقس")

WINDOW_SIZE = 7
EPOCHS = 15
BATCH_SIZE = 16

# -----------------------------
# اقتراح الملابس
# -----------------------------
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

# -----------------------------
# جلب الموقع من IP باستخدام ip-api.com
# -----------------------------
def get_location_by_ip(client_ip):
    if not client_ip:
        return None
    try:
        res = requests.get(f"http://ip-api.com/json/{client_ip}", timeout=6)
        res.raise_for_status()
        data = res.json()
        if data.get("status") != "success":
            return None
        lat = data.get("lat")
        lon = data.get("lon")
        city = data.get("city")
        timezone = data.get("timezone")
        if lat is not None and lon is not None:
            return {"lat": lat, "lon": lon, "timezone": timezone, "city": city}
    except:
        return None
    return None

# -----------------------------
# جلب بيانات الطقس من Open-Meteo
# -----------------------------
@st.cache_data(ttl=3600*6)
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

# -----------------------------
# جلب IP العميل تلقائي من المتصفح
# -----------------------------
if "client_ip" not in st.session_state:
    st.session_state.client_ip = None

# HTML + JS لجلب IP العميل
components.html("""
<script>
fetch('https://api.ipify.org?format=json')
.then(response => response.json())
.then(data => {
    const ip = data.ip;
    window.parent.postMessage({type:'client_ip', ip: ip}, "*");
});
</script>
""", height=0)

# استلام الرسائل من المتصفح
def on_message(message):
    if message.data.get("type") == "client_ip":
        st.session_state.client_ip = message.data.get("ip")

# تسجيل callback
components.html("""
<script>
window.addEventListener('message', function(event) {
    const data = event.data;
    if(data.type === 'client_ip'){
        const ipElem = document.getElementById('client_ip_holder');
        if(ipElem){
            ipElem.innerText = data.ip;
        }
    }
});
</script>
<div id="client_ip_holder" style="display:none"></div>
""", height=0)

# -----------------------------
# استخدم IP لجلب الموقع
# -----------------------------
if st.session_state.client_ip is None:
    st.info("⏳ جاري جلب IP العميل الخارجي...")
    st.stop()

client_ip = st.session_state.client_ip
loc = get_location_by_ip(client_ip)

if not loc:
    st.error("📌 تعذر جلب الموقع من IP العميل باستخدام ip-api.com")
    st.stop()

lat = loc["lat"]
lon = loc["lon"]
timezone = loc.get("timezone")
city = loc.get("city")
st.write(f"📍 **الموقع الحالي حسب IP:** {city}")

# -----------------------------
# التنبؤ بالطقس
# -----------------------------
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
