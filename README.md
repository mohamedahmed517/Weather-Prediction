# 🌦️ Weather Prediction App

This project is a **Web application** that predicts future weather conditions using a **trained LSTM deep learning model**.  
It fetches data, preprocesses it, and visualizes weather trends — all in a simple, user-friendly interface.

---

## 🚀 Features

- 📅 Predict weather for the upcoming days  
- 📊 Interactive charts and visualizations  
- 🌍 Easy-to-use interface
- 🤖 Powered by LSTM (Long Short-Term Memory) model for time-series forecasting  

---

## 🧠 Model Overview

The model is built using **TensorFlow Keras** with the following architecture:
- Input Layer  
- LSTM Layer(s)  
- Dense Output Layer  

The data is normalized using **MinMaxScaler** to enhance learning performance.

---

## 🌐 Usage

Enter the desired location and forecast period in the app interface.

The app will fetch and process the data, then display:

Predicted temperature trends

Weather statistics

Optional Arabic labels and right-to-left layout for Arabic content

---

## ⚙️ Install dependencies

pip install -r requirements.txt
