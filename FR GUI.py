import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import messagebox, simpledialog, scrolledtext
from PIL import Image, ImageTk
import pickle
import shutil
import datetime

# Paths and files
faces_path = "faces"
os.makedirs(faces_path, exist_ok=True)
users_file = "users.pkl"  # Stores username-password dict
log_file = "activity.log"

# Load Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Globals
cap = None
recognizing = False
label_dict = {}
current_user = None

# Load or create user database (username: password)
if os.path.exists(users_file):
    with open(users_file, "rb") as f:
        users_db = pickle.load(f)
else:
    users_db = {}
    with open(users_file, "wb") as f:
        pickle.dump(users_db, f)

def log_activity(text):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {text}\n"
    with open(log_file, "a") as f:
        f.write(log_entry)
    activity_text.config(state='normal')
    activity_text.insert(tk.END, log_entry)
    activity_text.see(tk.END)
    activity_text.config(state='disabled')

def detect_and_display():
    global cap, recognizing, frame_label, current_user
    if cap is None:
        cap = cv2.VideoCapture(0)

    ret, frame = cap.read()
    if not ret:
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))

        if recognizing:
            try:
                label, confidence = recognizer.predict(roi)
                if confidence < 100:
                    name = label_dict.get(label, "Unknown")
                else:
                    name = "Unknown"
            except:
                name = "Unknown"
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if name != "Unknown" and name == current_user:
                log_activity(f"User '{current_user}' recognized successfully.")

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    frame_label.imgtk = imgtk
    frame_label.configure(image=imgtk)
    frame_label.after(10, detect_and_display)

def start_recognition():
    global recognizing, label_dict, current_user
    if not current_user:
        messagebox.showerror("Error", "Please login first.")
        return
    try:
        recognizer.read("face_model.xml")
        with open("labels.pkl", "rb") as f:
            label_dict = pickle.load(f)
        recognizing = True
        log_activity(f"User '{current_user}' started face recognition.")
        detect_and_display()
    except:
        messagebox.showerror("Error", "Train the model first!")

def add_user():
    global current_user
    if not current_user:
        messagebox.showerror("Error", "Please login first.")
        return
    user_path = os.path.join(faces_path, current_user)
    os.makedirs(user_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))
            count += 1
            cv2.imwrite(f"{user_path}/{count}.jpg", face)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Adding User Images", frame)
        if cv2.waitKey(1) == 13 or count == 100:
            break

    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Success", f"Saved 100 images for user: {current_user}")
    log_activity(f"User '{current_user}' added face images.")

def train_model():
    global current_user
    if not current_user:
        messagebox.showerror("Error", "Please login first.")
        return
    data, labels = [], []
    label_map = {}
    current_label = 0

    for user in os.listdir(faces_path):
        user_folder = os.path.join(faces_path, user)
        if not os.path.isdir(user_folder):
            continue
        label_map[current_label] = user
        for img_name in os.listdir(user_folder):
            img_path = os.path.join(user_folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            data.append(np.asarray(img, dtype=np.uint8))
            labels.append(current_label)
        current_label += 1

    if not data:
        messagebox.showerror("Error", "No data found to train.")
        return

    recognizer.train(data, np.asarray(labels))
    recognizer.save("face_model.xml")
    with open("labels.pkl", "wb") as f:
        pickle.dump(label_map, f)
    messagebox.showinfo("Success", "Model trained successfully!")
    log_activity(f"User '{current_user}' trained the model.")

def logout():
    global recognizing, cap, current_user
    recognizing = False
    current_user = None
    if cap:
        cap.release()
    frame_label.configure(image="")
    messagebox.showinfo("Logout", "Logged out successfully.")
    login_frame.pack()
    main_frame.pack_forget()
    log_activity(f"User logged out.")

def delete_user():
    if not current_user:
        messagebox.showerror("Error", "Please login first.")
        return
    confirm = messagebox.askyesno("Confirm Delete", f"Delete user '{current_user}' and all data?")
    if confirm:
        user_path = os.path.join(faces_path, current_user)
        if os.path.exists(user_path):
            shutil.rmtree(user_path)
            messagebox.showinfo("Deleted", f"User '{current_user}' deleted.")
            log_activity(f"User '{current_user}' deleted their data.")
        else:
            messagebox.showerror("Error", f"User '{current_user}' data not found.")

def login():
    global current_user
    username = username_entry.get().strip()
    password = password_entry.get().strip()

    if username in users_db and users_db[username] == password:
        current_user = username
        messagebox.showinfo("Success", f"Welcome {username}!")
        log_activity(f"User '{username}' logged in successfully.")
        login_frame.pack_forget()
        main_frame.pack()
    else:
        messagebox.showerror("Error", "Invalid username or password.")
        log_activity(f"Failed login attempt for username '{username}'.")

def register():
    username = simpledialog.askstring("Register", "Enter new username:")
    if not username:
        return
    if username in users_db:
        messagebox.showerror("Error", "Username already exists.")
        return
    password = simpledialog.askstring("Register", "Enter password:", show='*')
    if not password:
        return
    users_db[username] = password
    with open(users_file, "wb") as f:
        pickle.dump(users_db, f)
    messagebox.showinfo("Success", f"User '{username}' registered successfully.")
    log_activity(f"User '{username}' registered.")

# GUI Setup

root = tk.Tk()
root.title("Face Recognition System with Login")
root.geometry("900x700")

# Login frame
login_frame = tk.Frame(root)
login_frame.pack(pady=100)

tk.Label(login_frame, text="Username:").grid(row=0, column=0, padx=5, pady=5)
username_entry = tk.Entry(login_frame)
username_entry.grid(row=0, column=1, padx=5, pady=5)

tk.Label(login_frame, text="Password:").grid(row=1, column=0, padx=5, pady=5)
password_entry = tk.Entry(login_frame, show="*")
password_entry.grid(row=1, column=1, padx=5, pady=5)

login_btn = tk.Button(login_frame, text="Login", command=login)
login_btn.grid(row=2, column=0, columnspan=2, pady=10)

register_btn = tk.Button(login_frame, text="Register", command=register)
register_btn.grid(row=3, column=0, columnspan=2, pady=5)

# Main frame (hidden until login)
main_frame = tk.Frame(root)

frame_label = tk.Label(main_frame)
frame_label.pack()

btn_frame = tk.Frame(main_frame)
btn_frame.pack(pady=10)

btn_add = tk.Button(btn_frame, text="Add Face Images", command=add_user)
btn_add.grid(row=0, column=0, padx=10)

btn_train = tk.Button(btn_frame, text="Train Model", command=train_model)
btn_train.grid(row=0, column=1, padx=10)

btn_recognize = tk.Button(btn_frame, text="Start Recognition", command=start_recognition)
btn_recognize.grid(row=0, column=2, padx=10)

btn_logout = tk.Button(btn_frame, text="Logout", command=logout)
btn_logout.grid(row=0, column=3, padx=10)

btn_delete = tk.Button(btn_frame, text="Delete User", command=delete_user)
btn_delete.grid(row=0, column=4, padx=10)

# Activity log display
tk.Label(main_frame, text="User Activity Log:").pack()
activity_text = scrolledtext.ScrolledText(main_frame, height=10, state='disabled')
activity_text.pack(fill=tk.X, padx=10, pady=5)

root.mainloop()
