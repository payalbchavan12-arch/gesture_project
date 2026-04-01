import cv2
import numpy as np
import tensorflow as tf
import pyttsx3
import time

# Load model
model = tf.keras.models.load_model("model.h5")

# Labels (same order as dataset folders)
labels = ["hello", "no", "yes"]

# Text-to-speech
engine = pyttsx3.init()

cap = cv2.VideoCapture(0)

last_prediction = ""
last_time = time.time()

while True:
    ret, frame = cap.read()

    # Draw rectangle (region of interest)
    cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)

    # Crop only hand area (VERY IMPORTANT)
    roi = frame[100:400, 100:400]

    # Preprocess
    img = cv2.resize(roi, (224, 224))
    img = img / 255.0
    img = np.reshape(img, (1, 224, 224, 3))

    # Prediction
    prediction = model.predict(img, verbose=0)
    index = np.argmax(prediction)
    text = labels[index]

    # Show text
    cv2.putText(frame, text, (120, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0,255,0), 2)

    cv2.imshow("Gesture Recognition", frame)

    # Speak only if prediction is stable
    if text != last_prediction and time.time() - last_time > 2:
        engine.say(text)
        engine.runAndWait()
        last_prediction = text
        last_time = time.time()

    # Exit
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()