import cv2
import numpy as np
import pickle

# Load model and labels
model = cv2.face.LBPHFaceRecognizer_create()
model.read("face_model.xml")

with open("labels.pkl", "rb") as f:
    label_dict = pickle.load(f)

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def face_detector(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return img, None
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
        return img, gray[y:y+h, x:x+w]
    return img, None

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    image, face = face_detector(frame)

    try:
        face = cv2.resize(face, (200, 200))
        result = model.predict(face)
        confidence = int((1 - result[1] / 300) * 100)

        if confidence > 75:
            name = label_dict[result[0]]
            cv2.putText(image, f"{name} ({confidence}%)", (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(image, "Unknown", (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    except:
        cv2.putText(image, "No face", (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Face Recognition", image)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
