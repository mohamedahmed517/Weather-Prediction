# streamlit.py
import requests
import numpy as np
import pandas as pd
import streamlit as st
from datetime import date, timedelta
import streamlit.components.v1 as components
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Input # type: ignore

st.set_page_config(page_title="توقعات الطقس", layout="centered")
st.title("تطبيق التنبؤ بالطقس")

WINDOW_SIZE = 7
EPOCHS = 15
BATCH_SIZE = 16

def suggest_outfit(temp, rain):
    if rain > 2: return "مطر خد شمسية"
    if temp < 10: return "برد قوي معطف"
    if temp < 18: return "بارد جاكيت"
    if temp < 26: return "لطيف تيشيرت"
    if temp < 32: return "دافئ تيشيرت خفيف"
    return "حر شورت"

def get_location(ip):
    try:
        r = requests.get(f"http://ip-api.com/json/{ip}", timeout=8)
        d = r.json()
        if d.get("status") == "success":
            return {"lat": d["lat"], "lon": d["lon"], "city": d.get("city", "غير معروف"), "tz": d.get("timezone", "UTC")}
    except:
        pass
    return None

@st.cache_data(ttl=3600*6)
def get_weather(lat, lon, tz):
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date=2020-01-01&end_date={date.today()}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max&timezone={tz}"
    try:
        return requests.get(url, timeout=15).json()
    except:
        return None

# ──────────────────────────────
# جلب الـ IP بـ 3 طرق (محلي + Cloud + Colab)
# ──────────────────────────────
client_ip = None

# 1. جرب من headers (Streamlit Cloud)
try:
    if hasattr(st, "context") and hasattr(st.context, "headers"):
        forwarded = st.context.headers.get("X-Forwarded-For")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()
except:
    pass

# 2. جرب من ipify.org (محلي أو Colab)
if not client_ip:
    try:
        client_ip = requests.get("https://api.ipify.org", timeout=5).text.strip()
    except:
        pass

# 3. لو لسه مفيش → نستخدم JavaScript (أضمن طريقة محلية)
if not client_ip:
    st.info("جاري جلب عنوان IP عبر المتصفح...")
    js = """
    <script>
    fetch('https://api.ipify.org?format=json')
      .then(r => r.json())
      .then(d => {
        document.getElementById('ip_holder').innerText = d.ip;
      })
      .catch(() => { document.getElementById('ip_holder').innerText = 'fallback_ip'; });
    </script>
    <div id="ip_holder" style="display:none;">جاري...</div>
    """
    components.html(js, height=0)
    # نعطي ثانيتين للـ JS
    import time
    time.sleep(2)
    # نجرب نقرأ الـ IP
    try:
        holder = components.html('<div id="ip_holder">جاري...</div>', height=0)
        # في الواقع هنا بنعتمد على rerun
        client_ip = "تم_بواسطة_JS"
    except:
        pass

# إذا لسه مفيش IP → نعطي زر "أعد المحاولة"
if not client_ip or client_ip in ["تم_بواسطة_JS", "fallback_ip"]:
    if st.button("أعد محاولة جلب IP"):
        st.rerun()
    st.stop()

st.success(f"تم جلب IP: `{client_ip}`")

# جلب الموقع
with st.spinner("جاري تحديد الموقع..."):
    loc = get_location(client_ip)

if not loc:
    st.error("تعذر تحديد الموقع. جرب تشغيل التطبيق على الإنترنت أو في Colab")
    st.stop()

lat, lon, city, tz = loc["lat"], loc["lon"], loc["city"], loc["tz"]
st.write(f"**المدينة:** {city}")
st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}), zoom=10)

# التنبؤ
days = st.slider("عدد الأيام", 1, 15, 3)
if st.button("ابدأ التنبؤ", type="primary"):
    with st.spinner("جاري تدريب النموذج..."):
        data = get_weather(lat, lon, tz)
        if not data or "daily" not in data:
            st.error("فشل في جلب بيانات الطقس")
            st.stop()

        df = pd.DataFrame(data["daily"])
        df["temp"] = (df["temperature_2m_max"] + df["temperature_2m_min"]) / 2
        df = df[["temp", "precipitation_sum", "windspeed_10m_max"]].astype(float)

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df.values)

        X, y = [], []
        for i in range(len(scaled) - WINDOW_SIZE):
            X.append(scaled[i:i+WINDOW_SIZE])
            y.append(scaled[i+WINDOW_SIZE, 0])
        X, y = np.array(X), np.array(y)

        model = Sequential([Input((WINDOW_SIZE, 3)), LSTM(64, return_sequences=True), LSTM(32), Dense(1)])
        model.compile("adam", "mse")
        model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

        last = scaled[-WINDOW_SIZE:]
        preds = []
        for _ in range(days):
            p = model.predict(last.reshape(1, WINDOW_SIZE, 3), verbose=0)[0][0]
            temp = scaler.inverse_transform([[p, 0, 0]])[0][0]
            preds.append(temp)
            last = np.vstack([last[1:], [p, last[-1,1], last[-1,2]]])

        rain = df["precipitation_sum"].iloc[-1]
        res = []
        for i, t in enumerate(preds, 1):
            d = (date.today() + timedelta(days=i)).strftime("%d/%m")
            outfit = suggest_outfit(t, rain)
            res.append({"التاريخ": d, "الحرارة": f"{t:.1f}°C", "الملابس": outfit})

        st.success("تم التنبؤ!")
        st.table(pd.DataFrame(res))
