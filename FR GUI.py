import tkinter as tk
from tkinter import messagebox, simpledialog
import cv2
import os
import numpy as np
import pickle
import pyttsx3
import threading
from datetime import datetime

# Initialize face detector and recognizer
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Create directories if they don't exist
if not os.path.exists('faces'):
    os.makedirs('faces')
if not os.path.exists('user_data'):
    os.makedirs('user_data')

# User database
users = {}
if os.path.exists('user_data/users.pkl'):
    with open('user_data/users.pkl', 'rb') as f:
        users = pickle.load(f)

def speak(text):
    def _speak():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=_speak).start()

class FaceRecognitionApp:
    def _init_(self, root):
        self.root = root
        self.root.title("Secure Face Recognition System")
        self.root.geometry("800x600")
        
        self.current_user = None
        self.current_frame = None
        self.door_status = "LOCKED"
        self.recognizer_trained = False
        
        # Load or train recognizer at startup
        self.load_or_train_recognizer()
        self.show_main_page()
    
    def load_or_train_recognizer(self):
        if os.path.exists('user_data/face_recognizer.yml'):
            recognizer.read('user_data/face_recognizer.yml')
            self.recognizer_trained = True
        else:
            self.train_recognizer()
    
    def clear_frame(self):
        if self.current_frame:
            self.current_frame.destroy()
    
    def show_main_page(self):
        self.clear_frame()
        self.current_frame = tk.Frame(self.root)
        self.current_frame.pack(expand=True, fill='both')
        
        tk.Label(self.current_frame, text="Security System", font=("Arial", 24)).pack(pady=30)
        
        tk.Button(self.current_frame, text="Register", font=("Arial", 16), 
                 command=self.show_register_page, width=20, height=2).pack(pady=20)
        
        tk.Button(self.current_frame, text="Login", font=("Arial", 16), 
                 command=self.show_login_page, width=20, height=2).pack(pady=20)
    
    def show_register_page(self):
        self.clear_frame()
        self.current_frame = tk.Frame(self.root)
        self.current_frame.pack(expand=True, fill='both')
        
        tk.Label(self.current_frame, text="Register New User", font=("Arial", 20)).pack(pady=20)
        
        self.username_var = tk.StringVar()
        tk.Label(self.current_frame, text="Username:", font=("Arial", 14)).pack()
        tk.Entry(self.current_frame, textvariable=self.username_var, font=("Arial", 14)).pack()
        
        tk.Button(self.current_frame, text="Capture Face", font=("Arial", 14), 
                 command=self.capture_face_for_registration).pack(pady=20)
        
        tk.Button(self.current_frame, text="Back", font=("Arial", 12), 
                 command=self.show_main_page).pack()
    
    def capture_face_for_registration(self):
        username = self.username_var.get().strip()
        if not username:
            messagebox.showerror("Error", "Please enter a username")
            return
        
        if username in users:
            messagebox.showerror("Error", "Username already exists")
            return
        
        cap = cv2.VideoCapture(0)
        face_samples = []
        sample_count = 0
        required_samples = 30  # Increased number of samples for better training
        
        def update_frame():
            nonlocal sample_count
            ret, frame = cap.read()
            if not ret:
                return
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100))
            
            for (x, y, w, h) in faces:
                if w > 100 and h > 100:  # Minimum face size
                    face_roi = gray[y:y+h, x:x+w]
                    
                    # Check face quality by looking at variance (blurry faces have low variance)
                    face_variance = cv2.Laplacian(face_roi, cv2.CV_64F).var()
                    
                    if face_variance > 50:  # Only accept clear faces
                        face_img = cv2.resize(face_roi, (200, 200))  # Resize for consistency
                        face_samples.append(face_img)
                        sample_count += 1
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, f"Samples: {sample_count}/{required_samples}", (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, f"Quality: {face_variance:.1f}", (x, y+h+20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    else:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.putText(frame, "Low Quality", (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Register Face - Press Q when done', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q') or sample_count >= required_samples:
                cap.release()
                cv2.destroyAllWindows()
                
                if len(face_samples) >= 25:  # Require at least 25 good samples
                    user_id = len(users) + 1
                    user_dir = f"faces/{user_id}"
                    os.makedirs(user_dir, exist_ok=True)
                    
                    # Save all face samples
                    for i, face in enumerate(face_samples):
                        cv2.imwrite(f"{user_dir}/{i}.png", face)
                    
                    # Add user to database
                    users[username] = {
                        'id': user_id,
                        'pin': None,
                        'registered_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'face_samples': len(face_samples)
                    }
                    self.save_users()
                    
                    # Train with new data
                    self.train_recognizer()
                    
                    messagebox.showinfo("Success", f"Face registered with {len(face_samples)} samples! Now set your PIN")
                    self.setup_pin_after_registration(username)
                else:
                    messagebox.showerror("Error", f"Not enough good quality face samples captured ({len(face_samples)}/25)")
        
        while cap.isOpened() and sample_count < required_samples:
            update_frame()
    
    def setup_pin_after_registration(self, username):
        pin = simpledialog.askstring("PIN Setup", "Set 4-digit PIN:", show='*')
        if pin and len(pin) == 4 and pin.isdigit():
            users[username]['pin'] = pin
            self.save_users()
            messagebox.showinfo("Success", "Registration complete!")
            speak(f"Registration successful for {username}")
            self.show_main_page()
        else:
            messagebox.showerror("Error", "PIN must be 4 digits")
            self.setup_pin_after_registration(username)
    
    def show_login_page(self):
        self.clear_frame()
        self.current_frame = tk.Frame(self.root)
        self.current_frame.pack(expand=True, fill='both')
        
        tk.Label(self.current_frame, text="Login", font=("Arial", 20)).pack(pady=20)
        
        self.login_username_var = tk.StringVar()
        tk.Label(self.current_frame, text="Username:", font=("Arial", 14)).pack()
        tk.Entry(self.current_frame, textvariable=self.login_username_var, font=("Arial", 14)).pack()
        
        tk.Button(self.current_frame, text="Login with Face", font=("Arial", 14), 
                 command=self.login_with_face).pack(pady=20)
        
        tk.Button(self.current_frame, text="Login with PIN", font=("Arial", 14), 
                 command=self.login_with_pin).pack(pady=20)
        
        tk.Button(self.current_frame, text="Back", font=("Arial", 12), 
                 command=self.show_main_page).pack()
    
    def login_with_face(self):
        if not self.recognizer_trained:
            messagebox.showerror("Error", "Face recognizer not trained yet")
            return
            
        username = self.login_username_var.get().strip()
        if not username:
            messagebox.showerror("Error", "Please enter username")
            return
        
        if username not in users:
            messagebox.showerror("Error", "User not found")
            return
        
        user_id = users[username]['id']
        cap = cv2.VideoCapture(0)
        recognized = False
        attempts = 0
        max_attempts = 3  # Reduced attempts for security
        confidence_threshold = 50  # Lower is better (more strict)
        
        def recognize_face():
            nonlocal recognized, attempts
            ret, frame = cap.read()
            if not ret:
                return
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100))
            
            for (x, y, w, h) in faces:
                if w > 100 and h > 100:  # Minimum face size
                    face_roi = gray[y:y+h, x:x+w]
                    face_img = cv2.resize(face_roi, (200, 200))  # Must match training size
                    
                    # Check face quality before recognition
                    face_variance = cv2.Laplacian(face_roi, cv2.CV_64F).var()
                    if face_variance < 50:  # Skip blurry faces
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.putText(frame, "Low Quality", (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        attempts += 1
                        continue
                    
                    id_, confidence = recognizer.predict(face_img)
                    
                    # Strict matching - only accept if confidence is very low (good match)
                    if id_ == user_id and confidence < confidence_threshold:
                        recognized = True
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, f"Welcome {username} ({confidence:.1f})", (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        attempts += 1
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.putText(frame, f"Attempt {attempts}/{max_attempts}", (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(frame, f"Confidence: {confidence:.1f}", (x, y+h+20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            cv2.imshow('Face Login - Press Q to cancel', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q') or recognized or attempts >= max_attempts:
                cap.release()
                cv2.destroyAllWindows()
                
                if recognized:
                    self.current_user = username
                    speak(f"Welcome {username}. Door is now open")
                    self.log_access(username, "FACE", True)
                    self.show_door_open_page()
                else:
                    messagebox.showerror("Error", "Face not recognized")
                    self.log_access(username, "FACE", False)
                    speak("Face recognition failed. Access denied")
        
        while cap.isOpened() and not recognized and attempts < max_attempts:
            recognize_face()
    
    def login_with_pin(self):
        username = self.login_username_var.get().strip()
        if not username:
            messagebox.showerror("Error", "Please enter username")
            return
        
        if username not in users:
            messagebox.showerror("Error", "User not found")
            return
        
        if not users[username]['pin']:
            messagebox.showerror("Error", "No PIN set for this user")
            return
        
        pin = simpledialog.askstring("PIN Login", "Enter 4-digit PIN:", show='*')
        if pin and pin == users[username]['pin']:
            self.current_user = username
            speak(f"Welcome {username}. Door is now open")
            self.log_access(username, "PIN", True)
            self.show_door_open_page()
        else:
            messagebox.showerror("Error", "Incorrect PIN")
            self.log_access(username, "PIN", False)
            speak("Incorrect PIN. Access denied")
    
    def show_door_open_page(self):
        self.clear_frame()
        self.current_frame = tk.Frame(self.root)
        self.current_frame.pack(expand=True, fill='both')
        
        tk.Label(self.current_frame, text="DOOR IS OPEN", 
                font=("Arial", 24, "bold"), fg="green").pack(pady=50)
        
        tk.Label(self.current_frame, text=f"Welcome {self.current_user}", 
                font=("Arial", 18)).pack(pady=10)
        
        tk.Button(self.current_frame, text="Lock Door", font=("Arial", 16), 
                 command=self.lock_door, width=20, height=2).pack(pady=20)
        
        self.door_status = "UNLOCKED"
        self.root.after(10000, self.lock_door)  # Auto lock after 10 sec
    
    def lock_door(self):
        if self.door_status == "UNLOCKED":
            self.door_status = "LOCKED"
            speak("Door is now locked")
            self.show_main_page()
    
    def train_recognizer(self):
        if not users:
            self.recognizer_trained = False
            return
            
        faces = []
        ids = []
        
        for user in users:
            user_id = users[user]['id']
            user_dir = f"faces/{user_id}"
            
            if os.path.exists(user_dir):
                for image_name in os.listdir(user_dir):
                    image_path = f"{user_dir}/{image_name}"
                    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (200, 200))  # Consistent size
                        faces.append(img)
                        ids.append(user_id)
        
        if faces and ids:
            recognizer.train(faces, np.array(ids))
            recognizer.save('user_data/face_recognizer.yml')
            self.recognizer_trained = True
            print(f"Recognizer trained with {len(faces)} samples for {len(set(ids))} users")
        else:
            self.recognizer_trained = False
            print("No training data available")
    
    def log_access(self, username, method, success):
        log_entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'method': method,
            'success': success
        }
        
        if 'access_log' not in users[username]:
            users[username]['access_log'] = []
        
        users[username]['access_log'].append(log_entry)
        self.save_users()
    
    def save_users(self):
        with open('user_data/users.pkl', 'wb') as f:
            pickle.dump(users, f)

if _name_ == "_main_":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
