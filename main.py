import cv2
import os
os.makedirs('static/face_snapshots', exist_ok=True)

import numpy as np
import os
import sqlite3
from datetime import datetime

# Model setup
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
gender_list = ['Male', 'Female']
gender_net = cv2.dnn.readNetFromCaffe(
    'deploy_gender.prototxt',
    'gender_net.caffemodel'
)

# Face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Output setup
if not os.path.exists('static'):
    os.makedirs('static')

# SQLite setup
conn = sqlite3.connect('face_logs.db')
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS detections (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, gender TEXT, x INTEGER, y INTEGER, width INTEGER, height INTEGER, image_path TEXT)")
conn.commit()

# Tracking variables
known_faces = []
total_people = 0
male_count = 0
female_count = 0

def is_new_face(x, y, w, h, known_faces, threshold=50):
    cx, cy = x + w // 2, y + h // 2
    for fx, fy in known_faces:
        if np.linalg.norm((cx - fx, cy - fy)) < threshold:
            return False
    known_faces.append((cx, cy))
    return True

prev_centers = {}
next_face_id = 0

count_in = 0
count_out = 0

cap = cv2.VideoCapture('rtsp://admin:keda123!!@192.168.18.67/')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    line_y = frame.shape[0] // 2
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (255, 0, 0), 2)

    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cx = x + w // 2
        cy = y + h // 2

        matched_id = None
        for face_id, (pcx, pcy) in prev_centers.items():
            if abs(cx - pcx) < 30 and abs(cy - pcy) < 30:
                matched_id = face_id
                break

        if matched_id is None:
            matched_id = next_face_id
            next_face_id += 1

        if matched_id in prev_centers:
            prev_cy = prev_centers[matched_id][1]
            if prev_cy < line_y and cy >= line_y:
                count_in += 1
            elif prev_cy > line_y and cy <= line_y:
                count_out += 1

        prev_centers[matched_id] = (cx, cy)

        if is_new_face(x, y, w, h, known_faces):
            face_img = frame[y:y+h, x:x+w].copy()
            blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]

            total_people += 1
            if gender == 'Male':
                male_count += 1
            else:
                female_count += 1

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            img_filename = f"static/face_snapshots/{timestamp}_{gender}.jpg"
            cv2.imwrite(img_filename, face_img)

            cursor.execute("INSERT INTO detections (timestamp, gender, x, y, width, height, image_path) VALUES (?, ?, ?, ?, ?, ?, ?)", 
                           (timestamp.replace('_', ' '), gender, int(x), int(y), int(w), int(h), img_filename))
            conn.commit()

        label = f"{gender}"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    count_text = f"Total: {total_people} | Male: {male_count} | Female: {female_count}"
    cv2.putText(frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv2.putText(frame, f"In: {count_in}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.putText(frame, f"Out: {count_out}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    cv2.imshow("People Counter + Gender + Logging", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
conn.close()
cv2.destroyAllWindows()
