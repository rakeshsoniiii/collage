import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog
from PIL import Image, ImageTk
import pickle
import threading
import time
import csv
from datetime import datetime
import pyttsx3

# Settings
FACE_DIR = "face_data"
UNIVERSAL_PASSWORD = "1234"

if not os.path.exists(FACE_DIR):
    os.makedirs(FACE_DIR)

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def log_login(name):
    with open("login_history.csv", "a", newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

class FaceUnlockSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Unlock System")
        self.root.geometry("800x700")

        self.cap = None
        self.running = False
        self.camera_index = 0
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
        tk.Button(root, text="Enter Password", command=self.check_password).pack(pady=5)
        tk.Button(root, text="Select Camera", command=self.select_camera).pack(pady=5)
        tk.Button(root, text="View/Delete Users", command=self.manage_users).pack(pady=5)
        tk.Button(root, text="Stop Camera", command=self.stop_camera).pack(pady=5)

    def load_labels_and_model(self):
        if os.path.exists("labels.pickle"):
            with open("labels.pickle", "rb") as f:
                self.labels = pickle.load(f)
        if os.path.exists("trainer.yml"):
            self.recognizer.read("trainer.yml")

    def select_camera(self):
        index = simpledialog.askinteger("Camera", "Enter camera index (0 for default):")
        if index is not None:
            self.camera_index = index
            messagebox.showinfo("Camera", f"Camera set to index {index}")

    def register_face(self):
        name = simpledialog.askstring("Face Registration", "Enter your name:")
        if not name:
            return
        user_path = os.path.join(FACE_DIR, name)
        os.makedirs(user_path, exist_ok=True)

        self.cap = cv2.VideoCapture(self.camera_index)
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
        self.cap = cv2.VideoCapture(self.camera_index)
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
                    speak(f"Welcome, {name}")
                    log_login(name)
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

    def check_password(self):
        pwd = simpledialog.askstring("Password Entry", "Enter the universal password:", show="*")
        if pwd == UNIVERSAL_PASSWORD:
            speak("Access granted")
            self.status.config(text="ACCESS GRANTED: PASSWORD", fg="green")
            log_login("Password User")
        else:
            self.status.config(text="ACCESS DENIED: PASSWORD", fg="red")
            speak("Access denied")

    def manage_users(self):
        win = tk.Toplevel(self.root)
        win.title("User Manager")
        users = os.listdir(FACE_DIR)
        for user in users:
            frame = tk.Frame(win)
            frame.pack(pady=2)
            tk.Label(frame, text=user).pack(side=tk.LEFT)
            tk.Button(frame, text="Delete", command=lambda u=user: self.delete_user(u, win)).pack(side=tk.RIGHT)

    def delete_user(self, username, window):
        if messagebox.askyesno("Delete User", f"Are you sure you want to delete user {username}?"):
            user_path = os.path.join(FACE_DIR, username)
            for file in os.listdir(user_path):
                os.remove(os.path.join(user_path, file))
            os.rmdir(user_path)
            messagebox.showinfo("Deleted", f"{username} deleted.")
            self.train_model()
            window.destroy()
            self.manage_users()

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
