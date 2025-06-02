import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
import cv2
import os
import numpy as np
import sqlite3
import pyttsx3
import threading
from datetime import datetime
from PIL import Image, ImageTk, ImageDraw, ImageFont
import hashlib
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

# Initialize face detector and recognizer
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Create directories if they don't exist
if not os.path.exists('faces'):
    os.makedirs('faces')

class Database:
    def __init__(self):
        self.conn = sqlite3.connect('user_data.db')
        self.create_tables()
    
    def create_tables(self):
        cursor = self.conn.cursor()
        
        # Users table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            pin_hash TEXT,
            registered_at TEXT NOT NULL,
            face_samples INTEGER,
            last_access TEXT
        )
        ''')
        
        # Access logs table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS access_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            method TEXT NOT NULL,
            success INTEGER NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')
        
        self.conn.commit()
    
    def add_user(self, username, pin_hash=None, face_samples=0):
        cursor = self.conn.cursor()
        try:
            cursor.execute('''
            INSERT INTO users (username, pin_hash, registered_at, face_samples)
            VALUES (?, ?, ?, ?)
            ''', (username, pin_hash, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), face_samples))
            user_id = cursor.lastrowid
            self.conn.commit()
            return user_id
        except sqlite3.IntegrityError:
            return None
    
    def get_user(self, username):
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        return cursor.fetchone()
    
    def get_user_by_id(self, user_id):
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
        return cursor.fetchone()
    
    def update_user_pin(self, username, pin_hash):
        cursor = self.conn.cursor()
        cursor.execute('''
        UPDATE users SET pin_hash = ? WHERE username = ?
        ''', (pin_hash, username))
        self.conn.commit()
    
    def update_user_last_access(self, username):
        cursor = self.conn.cursor()
        cursor.execute('''
        UPDATE users SET last_access = ? WHERE username = ?
        ''', (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), username))
        self.conn.commit()
    
    def add_access_log(self, user_id, method, success):
        cursor = self.conn.cursor()
        cursor.execute('''
        INSERT INTO access_logs (user_id, timestamp, method, success)
        VALUES (?, ?, ?, ?)
        ''', (user_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), method, success))
        self.conn.commit()
    
    def get_access_logs(self, user_id):
        cursor = self.conn.cursor()
        cursor.execute('''
        SELECT timestamp, method, success FROM access_logs 
        WHERE user_id = ? ORDER BY timestamp DESC
        ''', (user_id,))
        return cursor.fetchall()
    
    def get_all_users(self):
        cursor = self.conn.cursor()
        cursor.execute('SELECT id, username, registered_at, last_access, face_samples FROM users')
        return cursor.fetchall()
    
    def user_count(self):
        cursor = self.conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM users')
        return cursor.fetchone()[0]
    
    def delete_user(self, user_id):
        cursor = self.conn.cursor()
        try:
            # Delete access logs first (foreign key constraint)
            cursor.execute('DELETE FROM access_logs WHERE user_id = ?', (user_id,))
            # Delete user
            cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error deleting user: {e}")
            self.conn.rollback()
            return False
    
    def close(self):
        self.conn.close()

# Initialize database
db = Database()

def hash_pin(pin):
    """Hash a PIN for secure storage"""
    return hashlib.sha256(pin.encode()).hexdigest()

def verify_pin(pin_hash, pin):
    """Verify a PIN against its hash"""
    return pin_hash == hashlib.sha256(pin.encode()).hexdigest()

def speak(text):
    def _speak():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=_speak).start()

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Secure Face Recognition System")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f8ff')
        
        # Custom styles
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure styles
        self.style.configure('TButton', font=('Arial', 12), padding=10, 
                           background='#4b8bbe', foreground='white')
        self.style.map('TButton', 
                      background=[('active', '#3a6ea5'), ('pressed', '#2c5278')])
        
        self.style.configure('TLabel', font=('Arial', 12), background='#f0f8ff')
        self.style.configure('Header.TLabel', font=('Arial', 24, 'bold'), 
                           foreground='#2c3e50', background='#f0f8ff')
        self.style.configure('Success.TLabel', font=('Arial', 18, 'bold'), 
                           foreground='green', background='#f0f8ff')
        self.style.configure('Status.TLabel', font=('Arial', 10), 
                           foreground='gray', background='#f0f8ff')
        
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
    
    def create_icon(self, emoji, size):
        """Create an icon from an emoji character"""
        try:
            img = Image.new('RGBA', size, (0, 0, 0, 0))
            d = ImageDraw.Draw(img)
            
            # Try to use a larger font if available
            try:
                font = ImageFont.truetype("arial.ttf", size=min(size)//2)
            except:
                font = ImageFont.load_default()
            
            d.text((size[0]//2, size[1]//2), emoji, font=font, anchor="mm", fill="black")
            return ImageTk.PhotoImage(img)
        except Exception as e:
            print(f"Error creating icon: {e}")
            # Return a blank image if there's an error
            blank_img = Image.new('RGBA', size, (0, 0, 0, 0))
            return ImageTk.PhotoImage(blank_img)
    
    def show_main_page(self):
        self.clear_frame()
        self.current_frame = tk.Frame(self.root, bg='#f0f8ff')
        self.current_frame.pack(expand=True, fill='both', padx=40, pady=40)
        
        # Header
        header_frame = tk.Frame(self.current_frame, bg='#f0f8ff')
        header_frame.pack(pady=(0, 30))
        
        ttk.Label(header_frame, text="Secure Access System", style='Header.TLabel').pack()
        ttk.Label(header_frame, text="Face Recognition & PIN Security", 
                 style='Status.TLabel').pack()
        
        # Buttons container
        button_frame = tk.Frame(self.current_frame, bg='#f0f8ff')
        button_frame.pack(pady=20)
        
        # Main buttons with icons
        icon_size = (40, 40)
        
        # Register button
        self.register_icon = self.create_icon("📝", icon_size)
        ttk.Button(button_frame, text="Register New User", 
                  image=self.register_icon, compound=tk.LEFT,
                  command=self.show_register_page).grid(row=0, column=0, padx=15, pady=15, sticky='ew')
        
        # Login button
        self.login_icon = self.create_icon("🔑", icon_size)
        ttk.Button(button_frame, text="Login", 
                  image=self.login_icon, compound=tk.LEFT,
                  command=self.show_login_page).grid(row=1, column=0, padx=15, pady=15, sticky='ew')
        
        # View Users button
        self.users_icon = self.create_icon("👥", icon_size)
        ttk.Button(button_frame, text="View Users", 
                  image=self.users_icon, compound=tk.LEFT,
                  command=self.show_users_page).grid(row=2, column=0, padx=15, pady=15, sticky='ew')

        # View Unknown Faces button
        self.unknown_icon = self.create_icon("👻", icon_size) # Example icon
        ttk.Button(button_frame, text="View Unknown Faces",
                  image=self.unknown_icon, compound=tk.LEFT,
                  command=self.show_unknown_faces_page).grid(row=3, column=0, padx=15, pady=15, sticky='ew')
        
        # Status bar
        status_frame = tk.Frame(self.current_frame, bg='#e6f2ff', bd=1, relief=tk.SUNKEN)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(20, 0))
        
        ttk.Label(status_frame, 
                 text=f"System Status: {self.door_status} | Registered Users: {db.user_count()} | " + 
                      f"Face Recognition: {'Ready' if self.recognizer_trained else 'Not Trained'}",
                 style='Status.TLabel').pack(pady=5)
    
    def show_register_page(self):
        self.clear_frame()
        self.current_frame = tk.Frame(self.root, bg='#f0f8ff')
        self.current_frame.pack(expand=True, fill='both', padx=40, pady=40)
        
        # Header
        header_frame = tk.Frame(self.current_frame, bg='#f0f8ff')
        header_frame.pack(pady=(0, 30))
        
        ttk.Label(header_frame, text="Register New User", style='Header.TLabel').pack()
        ttk.Label(header_frame, text="Create a new security profile", 
                 style='Status.TLabel').pack()
        
        # Form frame
        form_frame = tk.Frame(self.current_frame, bg='#f0f8ff')
        form_frame.pack(pady=20)
        
        self.username_var = tk.StringVar()
        ttk.Label(form_frame, text="Username:").grid(row=0, column=0, padx=10, pady=10, sticky='e')
        ttk.Entry(form_frame, textvariable=self.username_var, font=('Arial', 12)).grid(row=0, column=1, padx=10, pady=10)
        
        # Button frame
        button_frame = tk.Frame(self.current_frame, bg='#f0f8ff')
        button_frame.pack(pady=30)
        
        self.capture_icon = self.create_icon("📷", (30, 30))
        ttk.Button(button_frame, text="Capture Face", 
                  image=self.capture_icon, compound=tk.LEFT,
                  command=self.capture_face_for_registration).pack(side=tk.LEFT, padx=10)
        
        self.back_icon = self.create_icon("⬅", (30, 30))
        ttk.Button(button_frame, text="Back to Main", 
                  image=self.back_icon, compound=tk.LEFT,
                  command=self.show_main_page).pack(side=tk.LEFT, padx=10)
        
        # Instructions
        ttk.Label(self.current_frame, 
                 text="Note: Face capture requires good lighting and clear front-facing view",
                 style='Status.TLabel').pack(side=tk.BOTTOM, pady=10)
    
    def capture_face_for_registration(self):
        username = self.username_var.get().strip()
        if not username:
            messagebox.showerror("Error", "Please enter a username")
            return
        
        if db.get_user(username):
            messagebox.showerror("Error", "Username already exists")
            return
        
        cap = cv2.VideoCapture(0)
        face_samples = []
        sample_count = 0
        required_samples = 30
        face_already_registered = False
        
        # Function to check if face already exists in the system
        def check_face_exists(face_img):
            if not self.recognizer_trained:
                return False, None
                
            id_, confidence = recognizer.predict(face_img)
            if confidence < 50:  # If confidence is high (value is low)
                user = db.get_user_by_id(id_)
                if user:
                    return True, user[1]  # Return True and the username
            return False, None
        
        def update_frame():
            nonlocal sample_count, face_already_registered
            ret, frame = cap.read()
            if not ret:
                return
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100))
            
            for (x, y, w, h) in faces:
                if w > 100 and h > 100:
                    face_roi = gray[y:y+h, x:x+w]
                    face_variance = cv2.Laplacian(face_roi, cv2.CV_64F).var()
                    
                    if face_variance > 50:
                        face_img = cv2.resize(face_roi, (200, 200))
                        
                        # Check if this face is already registered
                        if sample_count == 0:  # Only check on the first good sample
                            exists, existing_username = check_face_exists(face_img)
                            if exists:
                                face_already_registered = True
                                cap.release()
                                cv2.destroyAllWindows()
                                messagebox.showerror("Error", f"This face is already registered under username '{existing_username}'")
                                return
                        
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
                
                if len(face_samples) >= 25:
                    user_id = db.add_user(username, face_samples=len(face_samples))
                    if user_id:
                        user_dir = f"faces/{user_id}"
                        os.makedirs(user_dir, exist_ok=True)
                        
                        for i, face in enumerate(face_samples):
                            cv2.imwrite(f"{user_dir}/{i}.png", face)
                        
                        self.train_recognizer()
                        
                        messagebox.showinfo("Success", f"Face registered with {len(face_samples)} samples! Now set your PIN")
                        self.setup_pin_after_registration(username)
                else:
                    messagebox.showerror("Error", f"Not enough good quality face samples captured ({len(face_samples)}/25)")
        
        while cap.isOpened() and sample_count < required_samples and not face_already_registered:
            update_frame()
    
    def setup_pin_after_registration(self, username):
        pin = simpledialog.askstring("PIN Setup", "Set 4-digit PIN:", show='*')
        if pin and len(pin) == 4 and pin.isdigit():
            pin_hash = hash_pin(pin)
            db.update_user_pin(username, pin_hash)
            messagebox.showinfo("Success", "Registration complete!")
            speak(f"Registration successful for {username}")
            self.show_main_page()
        else:
            messagebox.showerror("Error", "PIN must be 4 digits")
            self.setup_pin_after_registration(username)
    
    def show_login_page(self):
        self.clear_frame()
        self.current_frame = tk.Frame(self.root, bg='#f0f8ff')
        self.current_frame.pack(expand=True, fill='both', padx=40, pady=40)
        
        # Header
        header_frame = tk.Frame(self.current_frame, bg='#f0f8ff')
        header_frame.pack(pady=(0, 30))
        
        ttk.Label(header_frame, text="System Login", style='Header.TLabel').pack()
        ttk.Label(header_frame, text="Choose your authentication method", 
                 style='Status.TLabel').pack()
        
        # Form frame for PIN login only
        form_frame = tk.Frame(self.current_frame, bg='#f0f8ff')
        form_frame.pack(pady=20)
        
        self.login_username_var = tk.StringVar()
        ttk.Label(form_frame, text="Username (for PIN only):").grid(row=0, column=0, padx=10, pady=10, sticky='e')
        ttk.Entry(form_frame, textvariable=self.login_username_var, font=('Arial', 12)).grid(row=0, column=1, padx=10, pady=10)
        
        # Button frame
        button_frame = tk.Frame(self.current_frame, bg='#f0f8ff')
        button_frame.pack(pady=30)
        
        self.face_icon = self.create_icon("👤", (30, 30))
        ttk.Button(button_frame, text="Unlock with Face", 
                  image=self.face_icon, compound=tk.LEFT,
                  command=self.login_with_face).pack(side=tk.LEFT, padx=10)
        
        self.pin_icon = self.create_icon("🔢", (30, 30))
        ttk.Button(button_frame, text="Unlock with PIN", 
                  image=self.pin_icon, compound=tk.LEFT,
                  command=self.login_with_pin).pack(side=tk.LEFT, padx=10)
        
        self.back_icon = self.create_icon("⬅", (30, 30))
        ttk.Button(button_frame, text="Back to Main", 
                  image=self.back_icon, compound=tk.LEFT,
                  command=self.show_main_page).pack(side=tk.LEFT, padx=10)
    
    def login_with_face(self):
        if not self.recognizer_trained:
            messagebox.showerror("Error", "Face recognizer not trained yet")
            return
        
        cap = cv2.VideoCapture(0)
        recognized = False
        attempts = 0
        max_attempts = 2
        confidence_threshold = 50
        unknown_face_captured = False
        recognized_user_id = None
        recognized_username = None
        
        def recognize_face():
            nonlocal recognized, attempts, unknown_face_captured, recognized_user_id, recognized_username
            ret, frame = cap.read()
            if not ret:
                return
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100))
            
            for (x, y, w, h) in faces:
                if w > 100 and h > 100:
                    face_roi = gray[y:y+h, x:x+w]
                    face_img = cv2.resize(face_roi, (200, 200))
                    
                    face_variance = cv2.Laplacian(face_roi, cv2.CV_64F).var()
                    if face_variance < 50:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.putText(frame, "Low Quality", (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        attempts += 1
                        continue
                    
                    id_, confidence = recognizer.predict(face_img)
                    
                    # Get user information based on recognized ID
                    user = db.get_user_by_id(id_)
                    
                    if user and confidence < confidence_threshold:
                        recognized = True
                        recognized_user_id = id_
                        recognized_username = user[1]  # Username is at index 1
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, f"Welcome {recognized_username} ({confidence:.1f})", (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        attempts += 1
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.putText(frame, f"Attempt {attempts}/{max_attempts}", (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(frame, f"Confidence: {confidence:.1f}", (x, y+h+20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        
                        # Save unknown face if it's the last attempt and not yet captured
                        if attempts >= max_attempts and not unknown_face_captured:
                            unknown_face_captured = True
                            # Ensure directory exists
                            os.makedirs("logs/unknown_faces", exist_ok=True)
                            # Save the unknown face with timestamp
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            unknown_face_path = f"logs/unknown_faces/unknown_{timestamp}.png"
                            cv2.imwrite(unknown_face_path, face_img)
                            print(f"Unknown face saved: {unknown_face_path}")
                            self.send_alert_email(unknown_face_path)
            
            cv2.imshow('Face Login - Press Q to cancel', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q') or recognized or attempts >= max_attempts:
                cap.release()
                cv2.destroyAllWindows()
                
                if recognized and recognized_username:
                    self.current_user = recognized_username
                    db.update_user_last_access(recognized_username)
                    speak(f"Welcome {recognized_username}. Door is now open")
                    db.add_access_log(recognized_user_id, "FACE", True)
                    self.show_door_open_page()
                else:
                    messagebox.showerror("Error", "Face not recognized")
                    # We don't know which user attempted login, so we can't log it to a specific user
                    speak("Face recognition failed. Access denied")
        
        while cap.isOpened() and not recognized and attempts < max_attempts:
            recognize_face()
    
    def login_with_pin(self):
        username = self.login_username_var.get().strip()
        if not username:
            messagebox.showerror("Error", "Please enter username")
            return
        
        user = db.get_user(username)
        if not user:
            messagebox.showerror("Error", "User not found")
            return
        
        if not user[2]:  # pin_hash
            messagebox.showerror("Error", "No PIN set for this user")
            return
        
        pin = simpledialog.askstring("PIN Login", "Enter 4-digit PIN:", show='*')
        if pin and len(pin) == 4 and pin.isdigit() and verify_pin(user[2], pin):
            self.current_user = username
            db.update_user_last_access(username)
            speak(f"Welcome {username}. Door is now open")
            db.add_access_log(user[0], "PIN", True)
            self.show_door_open_page()
        else:
            messagebox.showerror("Error", "Incorrect PIN")
            db.add_access_log(user[0], "PIN", False)
            speak("Incorrect PIN. Access denied")
    
    def show_door_open_page(self):
        self.clear_frame()
        self.current_frame = tk.Frame(self.root, bg='#f0f8ff')
        self.current_frame.pack(expand=True, fill='both', padx=40, pady=40)
        
        # Header
        header_frame = tk.Frame(self.current_frame, bg='#f0f8ff')
        header_frame.pack(pady=(0, 30))
        
        ttk.Label(header_frame, text="ACCESS GRANTED", style='Success.TLabel').pack()
        ttk.Label(header_frame, text=f"Welcome {self.current_user}", 
                 font=('Arial', 16)).pack()
        
        # Door status indicator
        status_frame = tk.Frame(self.current_frame, bg='#e6f7ff', bd=2, relief=tk.RIDGE)
        status_frame.pack(pady=30, ipadx=20, ipady=20)
        
        ttk.Label(status_frame, text="DOOR IS OPEN", 
                 font=('Arial', 24, 'bold'), foreground='green', 
                 background='#e6f7ff').pack(pady=20)
        
        # Button frame
        button_frame = tk.Frame(self.current_frame, bg='#f0f8ff')
        button_frame.pack(pady=20)
        
        self.lock_icon = self.create_icon("🔒", (30, 30))
        ttk.Button(button_frame, text="Lock Door", 
                  image=self.lock_icon, compound=tk.LEFT,
                  command=self.lock_door).pack(pady=10)
        
        self.door_status = "UNLOCKED"
        self.root.after(10000, self.lock_door)  # Auto lock after 10 sec
    
    def lock_door(self):
        if self.door_status == "UNLOCKED":
            self.door_status = "LOCKED"
            speak("Door is now locked")
            self.show_login_page()  # Go back to login page after locking
    
    def show_users_page(self):
        self.clear_frame()
        self.current_frame = tk.Frame(self.root, bg='#f0f8ff')
        self.current_frame.pack(expand=True, fill='both', padx=40, pady=40)
        
        # Header
        header_frame = tk.Frame(self.current_frame, bg='#f0f8ff')
        header_frame.pack(pady=(0, 20))
        
        ttk.Label(header_frame, text="Registered Users", style='Header.TLabel').pack()
        ttk.Label(header_frame, text="Manage system users", 
                 style='Status.TLabel').pack()
        
        # Create a frame for the Treeview and scrollbar
        tree_frame = tk.Frame(self.current_frame)
        tree_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        # Create a Treeview with scrollbar
        tree_scroll = ttk.Scrollbar(tree_frame)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.user_tree = ttk.Treeview(tree_frame, yscrollcommand=tree_scroll.set, 
                                    selectmode="browse")
        tree_scroll.config(command=self.user_tree.yview)
        
        # Define columns
        self.user_tree['columns'] = ("ID", "Username", "Registered At", "Last Access", "Face Samples")
        
        # Format columns
        self.user_tree.column("#0", width=0, stretch=tk.NO)
        self.user_tree.column("ID", anchor=tk.CENTER, width=50)
        self.user_tree.column("Username", anchor=tk.W, width=150)
        self.user_tree.column("Registered At", anchor=tk.W, width=180)
        self.user_tree.column("Last Access", anchor=tk.W, width=180)
        self.user_tree.column("Face Samples", anchor=tk.CENTER, width=100)
        
        # Create headings
        self.user_tree.heading("#0", text="", anchor=tk.W)
        self.user_tree.heading("ID", text="ID", anchor=tk.CENTER)
        self.user_tree.heading("Username", text="Username", anchor=tk.W)
        self.user_tree.heading("Registered At", text="Registered At", anchor=tk.W)
        self.user_tree.heading("Last Access", text="Last Access", anchor=tk.W)
        self.user_tree.heading("Face Samples", text="Samples", anchor=tk.CENTER)
        
        # Add data to treeview
        for user in db.get_all_users():
            self.user_tree.insert("", tk.END, values=(
                user[0],  # ID
                user[1],  # Username
                user[2],  # Registered At
                user[3] if user[3] else 'Never',  # Last Access
                user[4] if user[4] else 0  # Face Samples
            ))
        
        self.user_tree.pack(fill=tk.BOTH, expand=True)
        
        # Button frame
        button_frame = tk.Frame(self.current_frame, bg='#f0f8ff')
        button_frame.pack(pady=20)
        
        self.log_icon = self.create_icon("📋", (25, 25))
        ttk.Button(button_frame, text="View Access Log", 
                  image=self.log_icon, compound=tk.LEFT,
                  command=self.show_access_log).pack(side=tk.LEFT, padx=10)
        
        # Add delete user button (only visible for admin users)
        self.delete_icon = self.create_icon("🗑️", (25, 25))
        # Check if current user is admin or Cantor
        if self.current_user and self.current_user.lower() in ['admin', 'cantor']:
            ttk.Button(button_frame, text="Delete User", 
                      image=self.delete_icon, compound=tk.LEFT,
                      command=self.delete_user).pack(side=tk.LEFT, padx=10)
        
        self.back_icon = self.create_icon("⬅", (25, 25))
        ttk.Button(button_frame, text="Back to Main", 
                  image=self.back_icon, compound=tk.LEFT,
                  command=self.show_main_page).pack(side=tk.LEFT, padx=10)
    
    def show_access_log(self):
        selected = self.user_tree.focus()
        if not selected:
            messagebox.showwarning("Warning", "Please select a user first")
            return
        
        user_id = self.user_tree.item(selected)['values'][0]
        username = self.user_tree.item(selected)['values'][1]
        
        log_window = tk.Toplevel(self.root)
        log_window.title(f"Access Log for {username}")
        log_window.geometry("800x500")
        log_window.configure(bg='#f0f8ff')
        
        # Header
        ttk.Label(log_window, text=f"Access History for {username}", 
                 style='Header.TLabel').pack(pady=10)
        
        # Create a frame for the Treeview and scrollbar
        tree_frame = tk.Frame(log_window)
        tree_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        # Create a Treeview with scrollbar
        tree_scroll = ttk.Scrollbar(tree_frame)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        log_tree = ttk.Treeview(tree_frame, yscrollcommand=tree_scroll.set, 
                              selectmode="extended")
        tree_scroll.config(command=log_tree.yview)
        
        # Define columns
        log_tree['columns'] = ("Timestamp", "Method", "Success")
        
        # Format columns
        log_tree.column("#0", width=0, stretch=tk.NO)
        log_tree.column("Timestamp", anchor=tk.W, width=200)
        log_tree.column("Method", anchor=tk.CENTER, width=100)
        log_tree.column("Success", anchor=tk.CENTER, width=100)
        
        # Create headings
        log_tree.heading("#0", text="", anchor=tk.W)
        log_tree.heading("Timestamp", text="Timestamp", anchor=tk.W)
        log_tree.heading("Method", text="Method", anchor=tk.CENTER)
        log_tree.heading("Success", text="Success", anchor=tk.CENTER)
        
        # Add data to treeview
        for log in db.get_access_logs(user_id):
            success_text = "✅" if log[2] else "❌"
            log_tree.insert("", tk.END, values=(
                log[0],  # Timestamp
                log[1],  # Method
                success_text
            ))
        
        log_tree.pack(fill=tk.BOTH, expand=True)
        
        # Close button
        ttk.Button(log_window, text="Close", 
                  command=log_window.destroy).pack(pady=10)
    
    def delete_user(self):
        # Check if current user has permission (admin or Cantor)
        if not self.current_user or self.current_user.lower() not in ['admin', 'cantor']:
            messagebox.showerror("Error", "You don't have permission to delete users")
            return
        
        selected = self.user_tree.focus()
        if not selected:
            messagebox.showwarning("Warning", "Please select a user to delete")
            return
        
        user_id = self.user_tree.item(selected)['values'][0]
        username = self.user_tree.item(selected)['values'][1]
        
        # Don't allow deleting the current user or admin accounts
        if username.lower() in ['admin', 'cantor'] or username == self.current_user:
            messagebox.showerror("Error", "Cannot delete this user")
            return
        
        # Confirm deletion
        confirm = messagebox.askyesno("Confirm Deletion", 
                                    f"Are you sure you want to delete user '{username}'?\n\nThis will remove all user data and cannot be undone.")
        
        if confirm:
            # Delete user's face samples
            user_dir = f"faces/{user_id}"
            if os.path.exists(user_dir):
                for file in os.listdir(user_dir):
                    os.remove(f"{user_dir}/{file}")
                os.rmdir(user_dir)
            
            # Delete user from database
            success = db.delete_user(user_id)
            
            # Retrain recognizer
            self.train_recognizer()
            
            # Refresh user list
            messagebox.showinfo("Success", f"User '{username}' has been deleted")
            self.show_users_page()
    
    def send_alert_email(self, image_path=None):
        # Gmail SMTP settings
        HOST = "smtp.gmail.com"
        PORT = 587

        FROM_EMAIL = "xxxxxxxxx@gmail.com"    # enter sender email 🤙🤙👅👅
        TO_EMAIL = "xxxxxxxxxx@gmail.com" #  enter reserver email 🤙🤙👅
        APP_PASSWORD = "xxxxxxxxxx"  # 16-char app password from Google

        # Create the email message
        message = MIMEMultipart("alternative")
        message["Subject"] = "Smart Door Lock Alert: Unknown Face Detected at " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message["From"] = FROM_EMAIL
        message["To"] = TO_EMAIL

        # Email content (plain and HTML versions)
        text = f"""\
        Hello,

        An unknown face was detected attempting to access the smart door lock system.

        Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Location: Main Entrance
        Action Taken: Access Denied

        Please review the attached image if available.

        Regards,
        Smart Lock Bot
        """

        html = f"""\
        <html>
          <body>
            <h2>Smart Door Lock Notification</h2>
            <p>An unknown face was detected attempting to access the <i>smart door lock system</i>.</p>
            <p><b>Timestamp:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
               <b>Location:</b> Main Entrance<br>
               <b>Action Taken:</b> Access Denied</p>
            <p>Please review the attached image if available.</p>
            <p>Regards,<br>Smart Lock Bot</p>
          </body>
        </html>
        """

        # Attach both plain and HTML versions
        part1 = MIMEText(text, "plain")
        part2 = MIMEText(html, "html")
        message.attach(part1)
        message.attach(part2)

        # Attach the unknown face image if available
        if image_path and os.path.exists(image_path):
            with open(image_path, 'rb') as fp:
                img = MIMEImage(fp.read())
            img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(image_path))
            message.attach(img)

        # Send email
        try:
            smtp = smtplib.SMTP(HOST, PORT)
            smtp.ehlo()
            smtp.starttls()
            smtp.login(FROM_EMAIL, APP_PASSWORD)
            smtp.sendmail(FROM_EMAIL, TO_EMAIL, message.as_string())
            smtp.quit()
            print("[+] Security alert email sent successfully.")
        except Exception as e:
            print(f"[!] Failed to send security alert email: {e}")

    def train_recognizer(self):
        users = db.get_all_users()
        if not users:
            self.recognizer_trained = False
            return
            
        faces = []
        ids = []
        
        for user in users:
            user_id = user[0]
            user_dir = f"faces/{user_id}"
            
            if os.path.exists(user_dir):
                for image_name in os.listdir(user_dir):
                    image_path = f"{user_dir}/{image_name}"
                    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (200, 200))
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

    def show_unknown_faces_page(self):
        self.clear_frame()
        self.current_frame = tk.Frame(self.root, bg='#f0f8ff')
        self.current_frame.pack(expand=True, fill='both', padx=40, pady=40)

        ttk.Label(self.current_frame, text="Unknown Faces Log", style='Header.TLabel').pack(pady=(0, 20))

        canvas_frame = tk.Frame(self.current_frame, bg='#f0f8ff')
        canvas_frame.pack(fill='both', expand=True)

        canvas = tk.Canvas(canvas_frame, bg='#f0f8ff')
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, style='TFrame') # Use TFrame for styling if needed

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        unknown_faces_dir = "logs/unknown_faces"
        if not os.path.exists(unknown_faces_dir) or not os.listdir(unknown_faces_dir):
            ttk.Label(scrollable_frame, text="No unknown faces recorded.", style='Status.TLabel').pack(pady=20)
        else:
            row, col = 0, 0
            max_cols = 4 # Adjust as needed
            for i, filename in enumerate(os.listdir(unknown_faces_dir)):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        img_path = os.path.join(unknown_faces_dir, filename)
                        img = Image.open(img_path)
                        img.thumbnail((150, 150))  # Resize for display
                        photo_img = ImageTk.PhotoImage(img)
                        
                        img_label = ttk.Label(scrollable_frame, image=photo_img)
                        img_label.image = photo_img # Keep a reference!
                        img_label.grid(row=row, column=col, padx=10, pady=10)
                        
                        filename_label = ttk.Label(scrollable_frame, text=filename, style='Status.TLabel')
                        filename_label.grid(row=row+1, column=col, padx=10, pady=2)
                        
                        col += 1
                        if col >= max_cols:
                            col = 0
                            row += 2 # Increment by 2 for image and label
                    except Exception as e:
                        print(f"Error loading image {filename}: {e}")
        
        # Back button
        button_frame = tk.Frame(self.current_frame, bg='#f0f8ff') # Ensure this frame is separate
        button_frame.pack(pady=20, side=tk.BOTTOM)
        self.back_icon_unknown = self.create_icon("⬅", (25, 25))
        ttk.Button(button_frame, text="Back to Main", 
                  image=self.back_icon_unknown, compound=tk.LEFT,
                  command=self.show_main_page).pack()

if __name__ == "__main__":

    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
    db.close()
