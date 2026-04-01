from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pyttsx3
import time
import threading

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("model.h5")
labels = ["hello", "no", "yes"]

engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

last_prediction = ""
last_time = time.time()
current_text = "None"
camera_on = False


def generate_frames():
    global last_prediction, last_time, current_text, camera_on

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        if not camera_on:
            cap.release()
            break

        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:

                h, w, _ = frame.shape
                x_list, y_list = [], []

                for lm in hand_landmarks.landmark:
                    x_list.append(int(lm.x * w))
                    y_list.append(int(lm.y * h))

                x_min, x_max = min(x_list), max(x_list)
                y_min, y_max = min(y_list), max(y_list)

                x_min -= 20
                y_min -= 20
                x_max += 20
                y_max += 20

                roi = frame[y_min:y_max, x_min:x_max]

                if roi.size != 0:
                    img = cv2.resize(roi, (224, 224))
                    img = img / 255.0
                    img = np.reshape(img, (1, 224, 224, 3))

                    prediction = model.predict(img, verbose=0)
                    index = np.argmax(prediction)
                    text = labels[index]

                    current_text = text

                    cv2.putText(frame, text, (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0,255,0), 2)

                    if text != last_prediction and time.time() - last_time > 2:
                        threading.Thread(target=speak, args=(text,)).start()
                        last_prediction = text
                        last_time = time.time()

                    cv2.rectangle(frame, (x_min, y_min),
                                  (x_max, y_max), (0,255,0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start')
def start():
    global camera_on
    camera_on = True
    return jsonify({"status": "started"})


@app.route('/stop')
def stop():
    global camera_on, current_text
    camera_on = False
    current_text = "None"
    return jsonify({"status": "stopped"})


@app.route('/get_text')
def get_text():
    return jsonify({"text": current_text})


if __name__ == "__main__":
    app.run(debug=True)