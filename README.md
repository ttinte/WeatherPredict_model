## 📁 Project Structure

```text
AIoT-WEATHER-Station/
│
├── ai_server/
│   ├── server.py              # Flask API server
│   ├── model.h5               # Trained LSTM model
│   ├── scaler.pkl             # Data scaler
│   └── requirements.txt       # Python dependencies
│
├── WeatherPredictTraining/
│   ├── 01_check_timeseries.py # Missing value and outliers detection
│   ├── train_lstm_model.py    # Train model
│   ├── weather_fixed_pro.csv  # Dataset
│   ├── weather_lstm_model.h5  # Trained LSTM model (ignored)
│   ├── weather_scaler.pkl     # Data Scaler (ignored)
│   └── Predict_plot.png       # Prediction result
│
├── web_dashboard/
│   ├── index.html
│   ├── script.js
│   ├── style.css
│   ├── firebase.js
│   ├── map.js
│   ├── lang.js
│   └── logo.png
│
├── esp32_node/
│   └── main.ino               # ESP32 firmware
│
├── .gitignore
└── README.md
```
