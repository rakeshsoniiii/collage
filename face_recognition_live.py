import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import pyttsx3
import time

# --- TRAINING ---

data_path = 'G:/programings/vscode/Projects/6th sem final project/GrayScale image/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

Training_data, Labels = [], []

for i, file_name in enumerate(onlyfiles):
    image_path = join(data_path, file_name)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Skipping unreadable image: {image_path}")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    Training_data.append(np.asarray(gray, dtype=np.uint8))
    Labels.append(i)

Labels = np.asarray(Labels, dtype=np.int32)

model = cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(Training_data), np.asarray(Labels))
print("✅ Training complete!")

# Label dictionary (assumes filenames are like username.jpg)
label_dict = {i: onlyfiles[i].split('.')[0] for i in range(len(Labels))}

# --- FACE DETECTOR ---
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# --- SPEECH ENGINE ---
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty("voice", voices[0].id)
engine.setProperty("rate", 140)
engine.setProperty("volume", 1.0)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

# --- HEAD MOVEMENT TRACKING ---
face_positions = []

def calculate_movement(positions):
    if len(positions) < 2:
        return 0
    movement = 0
    for i in range(1, len(positions)):
        x1, y1 = positions[i - 1]
        x2, y2 = positions[i]
        movement += ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    return movement

# --- RECOGNITION LOOP ---
cap = cv2.VideoCapture(0)
recognized_count = 0
face_not_found_count = 0

print("📷 Starting camera. Please show your face and move your head slightly.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture frame from camera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    live_status = "Not Live"

    if len(faces) == 0:
        cv2.putText(frame, "Face not found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        face_not_found_count += 1
        if face_not_found_count > 30:
            speak("Face not found. Please try again.")
            face_not_found_count = 0
        face_positions.clear()
    else:
        face_not_found_count = 0
        for (x, y, w, h) in faces:
            center = (x + w // 2, y + h // 2)
            face_positions.append(center)
            if len(face_positions) > 20:
                face_positions.pop(0)

            movement = calculate_movement(face_positions)
            if movement > 20:
                live_status = "Live"
            else:
                live_status = "Not Live"

            roi = gray[y:y + h, x:x + w]
            roi = cv2.resize(roi, (200, 200))

            try:
                label, confidence = model.predict(roi)
                confidence_text = int(100 - confidence)

                if confidence < 100:
                    name = label_dict.get(label, "Unknown")
                    cv2.putText(frame, f"{name} ({live_status}) {confidence_text}%", (x, y - 10),
                                cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
                    if live_status == "Live":
                        recognized_count += 1
                    else:
                        recognized_count = 0
                else:
                    name = "Unknown"
                    cv2.putText(frame, f"Unknown ({live_status})", (x, y - 10),
                                cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
                    recognized_count = 0
            except:
                cv2.putText(frame, f"Error recognizing", (x, y - 10),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
                recognized_count = 0

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

    cv2.imshow('Face Recognition with Liveness', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 13:  # Enter
        break

    if recognized_count > 10:
        speak(f"Welcome {name}. Face recognition and liveness confirmed. Door is opening.")
        print(f"✅ User '{name}' recognized and live.")
        break

cap.release()
cv2.destroyAllWindows()
