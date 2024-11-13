import cv2
import urllib.request
import numpy as np
from ultralytics import YOLO
import paho.mqtt.client as mqtt
import json
import time
import firebase_admin
from firebase_admin import credentials, storage

# Load YOLO model
model = YOLO(r"D:\testPy\pythonProject1\runs\detect\train13\weights\best.pt")

url = 'http://192.168.137.60/cam-hi.jpg'

# MQTT configuration
mqtt_server = "50a989b3f1d24fbfa84a1e80f65a0e0a.s1.eu.hivemq.cloud"
mqtt_port = 8883
mqtt_username = "tuan.pa203636"
mqtt_password = "Matkhau123"
mqtt_topic = "door/control"

# Firebase configuration
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'esp32cam-e747d.appspot.com'
})
bucket = storage.bucket()

# MQTT client setup
client = mqtt.Client()
client.username_pw_set(mqtt_username, mqtt_password)
client.tls_set()  # Use default certificate
client.connect(mqtt_server, mqtt_port)


def send_mqtt_message(detection_type):
    message = {}

    if detection_type == "fire":
        message["fire_detected"] = True
    elif detection_type == "smoke":
        message["smoke_detected"] = True

    client.publish(mqtt_topic, json.dumps(message))


def upload_image_to_firebase(image):
    # Đặt tên tệp cố định cho ảnh
    filename = "detection_latest.jpg"
    image_path = f"./{filename}"  # Lưu vào thư mục hiện tại

    # Lưu ảnh vào đường dẫn tạm
    cv2.imwrite(image_path, image)

    # Tải lên Firebase Storage với tên tệp cố định
    blob = bucket.blob(f"images/{filename}")
    blob.upload_from_filename(image_path)
    print(f"Image {filename} uploaded to Firebase Storage.")



fire_detected_start = None
smoke_detected_start = None
detection_duration = 2  # Thời gian yêu cầu là 2 giây


def run():
    global fire_detected_start, smoke_detected_start
    cv2.namedWindow("live transmission + detection", cv2.WINDOW_AUTOSIZE)

    while True:
        try:
            # Lấy ảnh từ ESP32
            img_resp = urllib.request.urlopen(url)
            imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            im = cv2.imdecode(imgnp, -1)

            # Chuyển đổi sang không gian màu HSV
            hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

            # Điều chỉnh độ sáng và tương phản
            im = cv2.convertScaleAbs(im, alpha=1.3, beta=-30)

            # Làm mịn ảnh để giảm nhiễu
            im = cv2.GaussianBlur(im, (5, 5), 0)

            # Phát hiện đối tượng bằng YOLO
            results = model(im)

            fire_detected = False
            smoke_detected = False

            # Duyệt qua các kết quả phát hiện
            for r in results:
                classes_detected = r.boxes.cls.cpu().numpy()  # Lấy các lớp đối tượng đã phát hiện
                if 0 in classes_detected:
                    fire_detected = True
                elif 1 in classes_detected:
                    smoke_detected = True

            current_time = time.time()

            # Kiểm tra phát hiện lửa
            if fire_detected:
                if fire_detected_start is None:
                    fire_detected_start = current_time
                elif current_time - fire_detected_start >= detection_duration:
                    send_mqtt_message("fire")
                    upload_image_to_firebase(im)
                    fire_detected_start = None
            else:
                fire_detected_start = None

            # Kiểm tra phát hiện khói
            if smoke_detected:
                if smoke_detected_start is None:
                    smoke_detected_start = current_time
                elif current_time - smoke_detected_start >= detection_duration:
                    send_mqtt_message("smoke")
                    upload_image_to_firebase(im)
                    smoke_detected_start = None
            else:
                smoke_detected_start = None

            # Vẽ các bounding boxes lên ảnh
            annotated_img = results[0].plot()

            # Hiển thị ảnh với phát hiện đối tượng
            cv2.imshow('live transmission + detection', annotated_img)

            key = cv2.waitKey(5)
            if key == ord('q'):
                break
        except Exception as e:
            print(f"Error: {e}")
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    print("started")
    run()
