import cv2
import os

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def face_extractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None
    for (x, y, w, h) in faces:
        return gray[y:y+h, x:x+w]

username = input("Enter your username: ")
path = f"faces/{username}"
os.makedirs(path, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0
print("Capturing face images...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    face = face_extractor(frame)
    if face is not None:
        count += 1
        face = cv2.resize(face, (200, 200))
        cv2.imwrite(f"{path}/{count}.jpg", face)
        cv2.putText(frame, f"Face {count}/100", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Face Collector", frame)
    if cv2.waitKey(1) == 13 or count == 100:
        break

cap.release()
cv2.destroyAllWindows()
print("Data collection completed for", username)
