import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk
import pickle
import threading
import time

FACE_DIR = "face_data"

if not os.path.exists(FACE_DIR):
    os.makedirs(FACE_DIR)

class FaceUnlockSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Unlock System")
        self.root.geometry("700x600")

        self.cap = None
        self.running = False
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.labels = {}
        self.load_labels_and_model()

        self.video_label = tk.Label(self.root)
        self.video_label.pack()

        self.status = tk.Label(self.root, text="Status: Idle", font=("Arial", 16))
        self.status.pack(pady=10)

        tk.Button(root, text="Register Face", command=self.register_face).pack(pady=5)
        tk.Button(root, text="Unlock with Face", command=self.start_unlock).pack(pady=5)
        tk.Button(root, text="Stop", command=self.stop_camera).pack(pady=5)

    def load_labels_and_model(self):
        if os.path.exists("labels.pickle"):
            with open("labels.pickle", "rb") as f:
                self.labels = pickle.load(f)
        if os.path.exists("trainer.yml"):
            self.recognizer.read("trainer.yml")

    def register_face(self):
        name = simpledialog.askstring("Face Registration", "Enter your name:")
        if not name:
            return
        user_path = os.path.join(FACE_DIR, name)
        os.makedirs(user_path, exist_ok=True)

        self.cap = cv2.VideoCapture(0)
        count = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                count += 1
                face = gray[y:y+h, x:x+w]
                cv2.imwrite(os.path.join(user_path, f"{count}.jpg"), face)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                if count >= 30:
                    break
            cv2.imshow("Register Face", frame)
            if cv2.waitKey(1) & 0xFF == ord('q') or count >= 30:
                break

        self.cap.release()
        cv2.destroyAllWindows()
        messagebox.showinfo("Registration", f"Face registered for {name}")
        self.train_model()

    def train_model(self):
        faces = []
        labels = []
        label_ids = {}
        current_id = 0

        for root, dirs, files in os.walk(FACE_DIR):
            for name in dirs:
                path = os.path.join(root, name)
                if name not in label_ids:
                    label_ids[name] = current_id
                    current_id += 1
                label_id = label_ids[name]
                for img_name in os.listdir(path):
                    img_path = os.path.join(path, img_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    faces.append(img)
                    labels.append(label_id)

        with open("labels.pickle", "wb") as f:
            pickle.dump(label_ids, f)

        self.recognizer.train(faces, np.array(labels))
        self.recognizer.save("trainer.yml")
        self.labels = label_ids
        messagebox.showinfo("Training", "Model trained successfully!")

    def start_unlock(self):
        self.running = True
        self.cap = cv2.VideoCapture(0)
        threading.Thread(target=self.unlock_loop).start()

    def unlock_loop(self):
        reverse_labels = {v: k for k, v in self.labels.items()}

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.2, 5)

            unlocked = False

            for (x, y, w, h) in faces:
                roi = gray[y:y+h, x:x+w]
                id_, conf = self.recognizer.predict(roi)
                if conf < 70:
                    name = reverse_labels.get(id_, "Unknown")
                    self.status.config(text=f"ACCESS GRANTED: {name}", fg="green")
                    unlocked = True
                    cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                else:
                    self.status.config(text="ACCESS DENIED", fg="red")
                    cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            if unlocked:
                time.sleep(3)
                break

        self.cap.release()
        self.running = False

    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.video_label.config(image='')
        self.status.config(text="Status: Stopped", fg="black")

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceUnlockSystem(root)
    root.mainloop()
