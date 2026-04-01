import cv2
import os

gesture = input("Enter gesture name: ")
path = f"dataset/{gesture}"

if not os.path.exists(path):
    os.makedirs(path)

cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    cv2.imshow("Collecting Data", frame)

    # Automatically save every frame
    cv2.imwrite(f"{path}/{count}.jpg", frame)
    count += 1

    print(f"Saved: {count}")

    if count >= 200:
        break

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()