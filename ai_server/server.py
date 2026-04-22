from flask import Flask, jsonify, request
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import os
import joblib

import firebase_admin
from firebase_admin import credentials, db

app = Flask(__name__)
CORS(app)

# 1. Load model AI
model = tf.keras.models.load_model("weather_lstm_model.h5")

# 2. Load Scaler
scaler = joblib.load("weather_scaler.pkl")

# 3. Load API KEY From Render Environment Variables
API_KEY = os.environ.get("MY_API_KEY")

# 4. Initialize Firebase Admin SDK
cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://aiotnhom2-80e7a-default-rtdb.firebaseio.com/'
})

@app.route("/")
def home():
    return "Server AI dự báo thời tiết đang hoạt động!"

@app.route("/predict", methods=["GET", "POST"])
def predict():
    client_key = request.headers.get("x-api-key")

    # Debug Log:
    print(f"Key server đang có: '{API_KEY}'")
    print(f"Key Postman gửi lên: '{client_key}'")

    # Check API Key
    if client_key != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        ref = db.reference("weather_stations/Weather_station_1/readings")
        data_dict = ref.order_by_key().limit_to_last(2).get()

        if not data_dict:
            return jsonify({"error": "Không có dữ liệu trên Firebase"}), 400

        sequence = []
        
        for date_key in sorted(data_dict.keys()):
            day_data = data_dict[date_key]
            
            for time_key in sorted(day_data.keys()):
                val = day_data[time_key]
                try:
                    temp = float(val.get("temperature", 0.0))
                    hum  = float(val.get("humidity", 0.0))
                    pres = float(val.get("pressure", 0.0))
                    rain = float(val.get("rain", 0.0))
                    
                    sequence.append([temp, hum, pres, rain])
                except AttributeError:
                    continue

        if len(sequence) < 10:
            return jsonify({
                "error": f"Mới gom được {len(sequence)} bản ghi, cần ít nhất 10 để AI chạy."
            }), 400

        last_10_readings = sequence[-10:]
        scaled_input = scaler.transform(last_10_readings)
        x = np.array(scaled_input).reshape(1, 10, 4)
        scaled_pred = model.predict(x)
        real_pred = scaler.inverse_transform(scaled_pred)


        final_result = real_pred[0]

        if final_result[3] > 0.5:
            final_result[3] = 1.0
        else:
            final_result[3] = 0.0

        return jsonify({
            "status": "success",
            "prediction": [final_result.tolist()]
        })

    except Exception as e:
        # Bắt lỗi hệ thống để server không bị sập
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    # Tắt debug mode khi chạy trên Render
    app.run(host="0.0.0.0", port=port, debug=False)