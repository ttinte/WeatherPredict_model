from flask import Flask, jsonify, request
import tensorflow as tf
import numpy as np
import os

import firebase_admin
from firebase_admin import credentials, db

app = Flask(__name__)

# 1. Load model AI
model = tf.keras.models.load_model("weather_lstm_model.h5")

# 2. Lấy API KEY từ biến môi trường của Render
API_KEY = os.environ.get("MY_API_KEY")

# 3. Khởi tạo kết nối Firebase
# Đảm bảo bạn đã dán nội dung Service Account vào Secret Files trên Render với tên firebase_key.json
cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://aiotnhom2-default-rtdb.firebaseio.com/'
})

@app.route("/")
def home():
    return "Server AI dự báo thời tiết đang hoạt động!"

@app.route("/predict", methods=["GET", "POST"])
def predict():
    client_key = request.headers.get("x-api-key")

    # In ra log để dễ debug trên Render
    print(f"Key server đang có: '{API_KEY}'")
    print(f"Key Postman gửi lên: '{client_key}'")

    # Kiểm tra bảo mật
    if client_key != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        # 1. Trỏ vào thư mục 'readings' và lấy 2 ngày gần nhất
        # Việc lấy 2 ngày đảm bảo có đủ 10 bản ghi dù vừa mới bước qua ngày mới
        ref = db.reference("weather_stations/Weather_station_1/readings")
        data_dict = ref.order_by_key().limit_to_last(2).get()

        if not data_dict:
            return jsonify({"error": "Không có dữ liệu trên Firebase"}), 400

        sequence = []
        
        # 2. Lặp qua các thư mục Ngày (VD: 2026-04-03, 2026-04-04)
        for date_key in sorted(data_dict.keys()):
            day_data = data_dict[date_key]
            
            # 3. Lặp qua các mốc Giờ/Timestamp bên trong ngày đó
            # Sắp xếp theo Timestamp để đảm bảo AI nhận data đúng chiều thời gian
            for time_key in sorted(day_data.keys()):
                val = day_data[time_key]
                try:
                    temp = float(val.get("temperature", 0.0))
                    hum  = float(val.get("humidity", 0.0))
                    pres = float(val.get("pressure", 0.0))
                    rain = float(val.get("rain", 0.0))
                    
                    sequence.append([temp, hum, pres, rain])
                except AttributeError:
                    # Bỏ qua dòng này nếu cấu trúc dữ liệu bị lỗi
                    continue

        # 4. Kiểm tra xem sau khi gom lại đã đủ 10 bản ghi chưa
        if len(sequence) < 10:
            return jsonify({
                "error": f"Mới gom được {len(sequence)} bản ghi, cần ít nhất 10 để AI chạy."
            }), 400

        # 5. Cắt lấy đúng 10 bản ghi cuối cùng của mảng (10 mốc thời gian mới nhất)
        last_10_readings = sequence[-10:]

        # 6. Đưa vào Numpy array và reshape cho model LSTM
        # (1 sample, 10 timesteps, 4 features)
        x = np.array(last_10_readings).reshape(1, 10, 4)

        # 7. Chạy dự đoán
        pred = model.predict(x).tolist()

        return jsonify({
            "status": "success",
            "last_10_readings": last_10_readings,
            "prediction": pred
        })

    except Exception as e:
        # Bắt lỗi hệ thống để server không bị sập
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    # Tắt debug mode khi chạy trên Render
    app.run(host="0.0.0.0", port=port, debug=False)