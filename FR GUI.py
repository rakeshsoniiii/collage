import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
import cv2
import os
import numpy as np
import pickle
import pyttsx3
import threading
from datetime import datetime
from PIL import Image, ImageTk, ImageDraw, ImageFont

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
        
        # Status bar
        status_frame = tk.Frame(self.current_frame, bg='#e6f2ff', bd=1, relief=tk.SUNKEN)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(20, 0))
        
        ttk.Label(status_frame, 
                 text=f"System Status: {self.door_status} | Registered Users: {len(users)} | " + 
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
        
        if username in users:
            messagebox.showerror("Error", "Username already exists")
            return
        
        cap = cv2.VideoCapture(0)
        face_samples = []
        sample_count = 0
        required_samples = 30
        
        def update_frame():
            nonlocal sample_count
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
                    user_id = len(users) + 1
                    user_dir = f"faces/{user_id}"
                    os.makedirs(user_dir, exist_ok=True)
                    
                    for i, face in enumerate(face_samples):
                        cv2.imwrite(f"{user_dir}/{i}.png", face)
                    
                    users[username] = {
                        'id': user_id,
                        'pin': None,
                        'registered_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'face_samples': len(face_samples),
                        'last_access': None,
                        'access_log': []
                    }
                    self.save_users()
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
        self.current_frame = tk.Frame(self.root, bg='#f0f8ff')
        self.current_frame.pack(expand=True, fill='both', padx=40, pady=40)
        
        # Header
        header_frame = tk.Frame(self.current_frame, bg='#f0f8ff')
        header_frame.pack(pady=(0, 30))
        
        ttk.Label(header_frame, text="System Login", style='Header.TLabel').pack()
        ttk.Label(header_frame, text="Choose your authentication method", 
                 style='Status.TLabel').pack()
        
        # Form frame
        form_frame = tk.Frame(self.current_frame, bg='#f0f8ff')
        form_frame.pack(pady=20)
        
        self.login_username_var = tk.StringVar()
        ttk.Label(form_frame, text="Username:").grid(row=0, column=0, padx=10, pady=10, sticky='e')
        ttk.Entry(form_frame, textvariable=self.login_username_var, font=('Arial', 12)).grid(row=0, column=1, padx=10, pady=10)
        
        # Button frame
        button_frame = tk.Frame(self.current_frame, bg='#f0f8ff')
        button_frame.pack(pady=30)
        
        self.face_icon = self.create_icon("👤", (30, 30))
        ttk.Button(button_frame, text="Login with Face", 
                  image=self.face_icon, compound=tk.LEFT,
                  command=self.login_with_face).pack(side=tk.LEFT, padx=10)
        
        self.pin_icon = self.create_icon("🔢", (30, 30))
        ttk.Button(button_frame, text="Login with PIN", 
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
        max_attempts = 3
        confidence_threshold = 50
        
        def recognize_face():
            nonlocal recognized, attempts
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
                    users[username]['last_access'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.save_users()
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
            users[username]['last_access'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.save_users()
            speak(f"Welcome {username}. Door is now open")
            self.log_access(username, "PIN", True)
            self.show_door_open_page()
        else:
            messagebox.showerror("Error", "Incorrect PIN")
            self.log_access(username, "PIN", False)
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
        for username in users:
            user_data = users[username]
            self.user_tree.insert("", tk.END, values=(
                user_data['id'],
                username,
                user_data['registered_at'],
                user_data.get('last_access', 'Never'),
                user_data.get('face_samples', 0)
            ))
        
        self.user_tree.pack(fill=tk.BOTH, expand=True)
        
        # Button frame
        button_frame = tk.Frame(self.current_frame, bg='#f0f8ff')
        button_frame.pack(pady=20)
        
        self.log_icon = self.create_icon("📋", (25, 25))
        ttk.Button(button_frame, text="View Access Log", 
                  image=self.log_icon, compound=tk.LEFT,
                  command=self.show_access_log).pack(side=tk.LEFT, padx=10)
        
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
        if 'access_log' in users[username]:
            for log in reversed(users[username]['access_log']):  # Show most recent first
                success_text = "✅" if log['success'] else "❌"
                log_tree.insert("", tk.END, values=(
                    log['timestamp'],
                    log['method'],
                    success_text
                ))
        
        log_tree.pack(fill=tk.BOTH, expand=True)
        
        # Close button
        ttk.Button(log_window, text="Close", 
                  command=log_window.destroy).pack(pady=10)
    
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

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
