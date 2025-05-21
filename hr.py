import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import messagebox, simpledialog, scrolledtext, ttk
from PIL import Image, ImageTk
import pickle
import shutil
import datetime
import time
import threading

# Paths and files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
faces_path = os.path.join(BASE_DIR, "faces")
os.makedirs(faces_path, exist_ok=True)
users_file = os.path.join(BASE_DIR, "users.pkl")
log_file = os.path.join(BASE_DIR, "activity.log")
model_file = os.path.join(BASE_DIR, "face_model.xml")
labels_file = os.path.join(BASE_DIR, "labels.pkl")

# Load Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Globals
cap = None
recognizing = False
label_dict = {}
current_user = None
training_in_progress = False
capturing = False

# Load or initialize user database
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
    with open(log_file, "a", encoding='utf-8') as f:
        f.write(log_entry)
    activity_text.config(state='normal')
    activity_text.insert(tk.END, log_entry)
    activity_text.see(tk.END)
    activity_text.config(state='disabled')

def detect_and_display():
    global cap, recognizing, frame_label, current_user
    if cap is None:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    ret, frame = cap.read()
    if not ret:
        frame_label.after(10, detect_and_display)
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))

        if recognizing:
            try:
                label, confidence = recognizer.predict(roi)
                name = label_dict.get(label, "Unknown") if confidence < 85 else "Unknown"
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.putText(frame, f"{name} ({confidence:.1f}%)", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                if name != "Unknown" and name == current_user:
                    log_activity(f"User '{current_user}' recognized (confidence: {confidence:.1f}%).")
            except Exception as e:
                print(f"Recognition error: {e}")
                name = "Unknown"
                cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

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
    
    if not os.path.exists(model_file):
        messagebox.showerror("Error", "No trained model found. Please train the model first!")
        return
    
    try:
        recognizer.read(model_file)
        with open(labels_file, "rb") as f:
            label_dict = pickle.load(f)
        recognizing = True
        log_activity(f"User '{current_user}' started face recognition.")
        detect_and_display()
        recognition_btn.config(text="Stop Recognition", command=stop_recognition, style='Accent.TButton')
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load model: {str(e)}")

def stop_recognition():
    global recognizing
    recognizing = False
    recognition_btn.config(text="Start Recognition", command=start_recognition, style='TButton')
    log_activity(f"User '{current_user}' stopped face recognition.")

def add_user():
    global current_user, capturing
    if not current_user:
        messagebox.showerror("Error", "Please login first.")
        return
    
    if capturing:
        messagebox.showwarning("Warning", "Already capturing images!")
        return
    
    user_path = os.path.join(faces_path, current_user)
    os.makedirs(user_path, exist_ok=True)
    
    # Create a new window for capturing images
    capture_window = tk.Toplevel(root)
    capture_window.title(f"Adding Face Images for {current_user}")
    capture_window.geometry("800x600")
    capture_window.resizable(False, False)
    
    # Create frame for video display
    video_frame = tk.Frame(capture_window)
    video_frame.pack(pady=10)
    
    # Create label for displaying video
    video_label = tk.Label(video_frame)
    video_label.pack()
    
    # Create progress bar
    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(capture_window, variable=progress_var, maximum=100, length=400)
    progress_bar.pack(pady=10)
    
    # Create status label
    status_label = tk.Label(capture_window, text="Preparing to capture images...", 
                           font=('Helvetica', 10))
    status_label.pack(pady=5)
    
    # Create button to stop capturing
    stop_btn = ttk.Button(capture_window, text="Stop Capturing", 
                         command=lambda: stop_capturing(capture_window))
    stop_btn.pack(pady=10)
    
    # Start capturing process
    def capture_images():
        global capturing
        capturing = True
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        count = 0
        max_images = 100
        
        while count < max_images and capturing:
            ret, frame = cap.read()
            if not ret:
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 1:  # Only capture when exactly one face is detected
                (x, y, w, h) = faces[0]
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (200, 200))
                
                # Save every 5 frames to get varied expressions
                if count % 5 == 0:
                    cv2.imwrite(f"{user_path}/{current_user}_{count}.jpg", face)
                    count += 1
                    progress_var.set(count)
                    status_label.config(text=f"Captured {count}/{max_images} images")
                    capture_window.update()
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Display the frame
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.config(image=imgtk)
            capture_window.update()
            
            time.sleep(0.05)  # Small delay to allow UI updates
        
        cap.release()
        capturing = False
        
        if count >= max_images:
            messagebox.showinfo("Success", f"Successfully captured {max_images} images for user: {current_user}")
            log_activity(f"User '{current_user}' added face images.")
        else:
            log_activity(f"User '{current_user}' stopped capturing after {count} images.")
        
        capture_window.destroy()
    
    # Start the capturing process in a separate thread
    threading.Thread(target=capture_images, daemon=True).start()

def stop_capturing(window):
    global capturing
    capturing = False
    window.destroy()
    log_activity(f"User '{current_user}' stopped adding face images.")

def train_model():
    global current_user, training_in_progress
    if not current_user:
        messagebox.showerror("Error", "Please login first.")
        return
    
    if training_in_progress:
        messagebox.showwarning("Warning", "Training is already in progress.")
        return
    
    # Check if there are any users with face images
    has_data = False
    for user in os.listdir(faces_path):
        user_folder = os.path.join(faces_path, user)
        if os.path.isdir(user_folder) and len(os.listdir(user_folder)) > 0:
            has_data = True
            break
    
    if not has_data:
        messagebox.showerror("Error", "No face images found to train. Please add face images first.")
        return
    
    # Create a progress window
    progress_window = tk.Toplevel(root)
    progress_window.title("Training Model")
    progress_window.geometry("400x200")
    progress_window.resizable(False, False)
    
    tk.Label(progress_window, text="Training Face Recognition Model", 
            font=('Helvetica', 12, 'bold')).pack(pady=10)
    
    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=100)
    progress_bar.pack(pady=20, padx=20, fill=tk.X)
    
    status_label = tk.Label(progress_window, text="Preparing training data...")
    status_label.pack(pady=5)
    
    def training_thread():
        global training_in_progress
        training_in_progress = True
        
        try:
            data, labels = [], []
            label_map = {}
            current_label = 0
            
            # Get all user folders
            users = [d for d in os.listdir(faces_path) if os.path.isdir(os.path.join(faces_path, d))]
            total_users = len(users)
            
            for i, user in enumerate(users):
                user_folder = os.path.join(faces_path, user)
                if not os.path.isdir(user_folder):
                    continue
                
                label_map[current_label] = user
                image_files = [f for f in os.listdir(user_folder) if f.endswith(('.jpg', '.png'))]
                total_images = len(image_files)
                
                status_label.config(text=f"Processing {user} ({total_images} images)")
                progress_window.update()
                
                for j, img_name in enumerate(image_files):
                    img_path = os.path.join(user_folder, img_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        data.append(np.asarray(img, dtype=np.uint8))
                        labels.append(current_label)
                    
                    # Update progress
                    progress = ((i + (j+1)/total_images) / total_users) * 100
                    progress_var.set(progress)
                    progress_window.update()
                
                current_label += 1
            
            if not data:
                messagebox.showerror("Error", "No valid training data found.")
                progress_window.destroy()
                training_in_progress = False
                return
            
            status_label.config(text="Training the model...")
            progress_window.update()
            
            # Train the model with updated parameters for better accuracy
            recognizer.train(data, np.asarray(labels))
            recognizer.save(model_file)
            
            with open(labels_file, "wb") as f:
                pickle.dump(label_map, f)
            
            messagebox.showinfo("Success", "Model trained successfully!")
            log_activity(f"User '{current_user}' trained the model with {len(data)} samples.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {str(e)}")
            log_activity(f"Training failed: {str(e)}")
        finally:
            progress_window.destroy()
            training_in_progress = False
    
    # Start training in a separate thread
    threading.Thread(target=training_thread, daemon=True).start()

def logout():
    global recognizing, cap, current_user
    recognizing = False
    current_user = None
    if cap:
        cap.release()
        cap = None
    frame_label.configure(image="")
    messagebox.showinfo("Logout", "Logged out successfully.")
    login_frame.pack()
    main_frame.pack_forget()
    username_entry.delete(0, tk.END)
    password_entry.delete(0, tk.END)
    log_activity("User logged out.")

def delete_user():
    if not current_user:
        messagebox.showerror("Error", "Please login first.")
        return
    
    confirm = messagebox.askyesno("Confirm Delete", 
                                f"Are you sure you want to delete user '{current_user}' and all associated data?",
                                icon='warning')
    if confirm:
        user_path = os.path.join(faces_path, current_user)
        if os.path.exists(user_path):
            try:
                shutil.rmtree(user_path)
                messagebox.showinfo("Deleted", f"User '{current_user}' and all data deleted successfully.")
                log_activity(f"User '{current_user}' deleted their data.")
                
                # Check if this user exists in the label dictionary
                if os.path.exists(labels_file):
                    with open(labels_file, "rb") as f:
                        label_dict = pickle.load(f)
                    
                    # Find and remove the user from label dictionary
                    label_to_remove = None
                    for label, name in label_dict.items():
                        if name == current_user:
                            label_to_remove = label
                            break
                    
                    if label_to_remove is not None:
                        del label_dict[label_to_remove]
                        with open(labels_file, "wb") as f:
                            pickle.dump(label_dict, f)
                        
                        # Retrain the model if it exists
                        if os.path.exists(model_file):
                            train_model()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete user data: {str(e)}")
        else:
            messagebox.showerror("Error", f"No data found for user '{current_user}'")

def login():
    global current_user
    username = username_entry.get().strip()
    password = password_entry.get().strip()

    if not username or not password:
        messagebox.showerror("Error", "Username and password cannot be empty.")
        return

    if username in users_db and users_db[username] == password:
        current_user = username
        messagebox.showinfo("Success", f"Welcome {username}!")
        log_activity(f"User '{username}' logged in successfully.")
        login_frame.pack_forget()
        main_frame.pack()
        username_entry.delete(0, tk.END)
        password_entry.delete(0, tk.END)
    else:
        messagebox.showerror("Error", "Invalid username or password.")
        log_activity(f"Failed login attempt for username '{username}'.")

def register():
    # Create registration window
    register_window = tk.Toplevel(root)
    register_window.title("Register New User")
    register_window.geometry("400x300")
    register_window.resizable(False, False)
    
    # Registration form
    tk.Label(register_window, text="Register New User", font=('Helvetica', 16, 'bold')).pack(pady=10)
    
    tk.Label(register_window, text="Username:").pack()
    reg_username = tk.Entry(register_window)
    reg_username.pack(pady=5)
    
    tk.Label(register_window, text="Password:").pack()
    reg_password = tk.Entry(register_window, show="*")
    reg_password.pack(pady=5)
    
    tk.Label(register_window, text="Confirm Password:").pack()
    reg_confirm = tk.Entry(register_window, show="*")
    reg_confirm.pack(pady=5)
    
    def submit_registration():
        username = reg_username.get().strip()
        password = reg_password.get().strip()
        confirm = reg_confirm.get().strip()
        
        if not username or not password:
            messagebox.showerror("Error", "Username and password cannot be empty.")
            return
        
        if not username.isalnum():
            messagebox.showerror("Error", "Username can only contain letters and numbers.")
            return
        
        if username in users_db:
            messagebox.showerror("Error", "Username already exists.")
            return
        
        if len(password) < 4:
            messagebox.showerror("Error", "Password must be at least 4 characters long.")
            return
        
        if password != confirm:
            messagebox.showerror("Error", "Passwords do not match.")
            return
        
        users_db[username] = password
        with open(users_file, "wb") as f:
            pickle.dump(users_db, f)
        
        messagebox.showinfo("Success", f"User '{username}' registered successfully.")
        log_activity(f"New user '{username}' registered.")
        register_window.destroy()
    
    submit_btn = ttk.Button(register_window, text="Register", command=submit_registration)
    submit_btn.pack(pady=15)

def show_help():
    help_text = """Face Recognition System Help:

1. Register: Create a new account with username and password
2. Login: Access the system with your credentials
3. Add Face Images: Capture 100 images of your face
4. Train Model: Train the system to recognize your face
5. Start Recognition: Begin face recognition
6. Delete User: Remove your account and all data

For best results:
- Ensure good lighting when adding face images
- Capture images with different expressions
- Train the model after adding new face images
"""
    messagebox.showinfo("Help", help_text)

# GUI Setup with professional styling
root = tk.Tk()
root.title("Advanced Face Recognition System")
root.geometry("1000x800")
root.resizable(False, False)

# Color scheme
bg_color = "#f0f0f0"
accent_color = "#4a6fa5"
text_color = "#333333"
button_color = "#4a6fa5"
button_hover = "#3a5a80"
progress_color = "#4a6fa5"

# Apply styles
style = ttk.Style()
style.theme_use('clam')

# Configure styles
style.configure('TFrame', background=bg_color)
style.configure('TLabel', background=bg_color, foreground=text_color, font=('Helvetica', 11))
style.configure('TButton', font=('Helvetica', 11), padding=6, background=button_color, foreground='white')
style.map('TButton', 
          background=[('active', button_hover), ('!disabled', button_color)],
          foreground=[('!disabled', 'white')])
style.configure('Accent.TButton', background='#4CAF50', foreground='white')
style.configure('Warning.TButton', background='#f44336', foreground='white')
style.configure('TEntry', font=('Helvetica', 11), padding=5, fieldbackground='white', foreground=text_color)
style.configure('Title.TLabel', font=('Helvetica', 18, 'bold'))
style.configure('Horizontal.TProgressbar', background=progress_color)

# Login Frame
login_frame = ttk.Frame(root, style='TFrame')
login_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

ttk.Label(login_frame, text="Face Recognition System", style='Title.TLabel').grid(row=0, column=0, columnspan=2, pady=(0, 20))

ttk.Label(login_frame, text="Username:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
username_entry = ttk.Entry(login_frame)
username_entry.grid(row=1, column=1, padx=5, pady=5)

ttk.Label(login_frame, text="Password:").grid(row=2, column=0, sticky="e", padx=5, pady=5)
password_entry = ttk.Entry(login_frame, show="*")
password_entry.grid(row=2, column=1, padx=5, pady=5)

login_btn = ttk.Button(login_frame, text="Login", command=login, style='TButton')
login_btn.grid(row=3, column=0, columnspan=2, pady=10, sticky="ew")

register_btn = ttk.Button(login_frame, text="Register", command=register, style='TButton')
register_btn.grid(row=4, column=0, columnspan=2, pady=(0, 15), sticky="ew")

help_btn = ttk.Button(login_frame, text="Help", command=show_help, style='TButton')
help_btn.grid(row=5, column=0, columnspan=2, pady=(0, 10), sticky="ew")

# Main Frame
main_frame = ttk.Frame(root, style='TFrame')

# Header with user info
header_frame = ttk.Frame(main_frame, style='TFrame')
header_frame.pack(fill=tk.X, padx=10, pady=10)

ttk.Label(header_frame, text=f"Welcome, ", font=('Helvetica', 12)).pack(side=tk.LEFT)
user_label = ttk.Label(header_frame, text="", font=('Helvetica', 12, 'bold'))
user_label.pack(side=tk.LEFT)
logout_btn = ttk.Button(header_frame, text="Logout", command=logout, style='TButton')
logout_btn.pack(side=tk.RIGHT)

# Video display frame
video_container = ttk.Frame(main_frame, style='TFrame')
video_container.pack(pady=10)

frame_label = ttk.Label(video_container)
frame_label.pack()

# Buttons frame
btn_frame = ttk.Frame(main_frame, style='TFrame')
btn_frame.pack(pady=15)

# Create buttons with improved layout
button_specs = [
    ("Add Face Images", add_user, 'TButton'),
    ("Train Model", train_model, 'TButton'),
    ("Start Recognition", start_recognition, 'TButton'),
    ("Delete User", delete_user, 'Warning.TButton')
]

for i, (text, cmd, style_name) in enumerate(button_specs):
    btn = ttk.Button(btn_frame, text=text, command=cmd, style=style_name, width=20)
    btn.grid(row=0, column=i, padx=5, pady=5)

# Activity log
log_frame = ttk.Frame(main_frame, style='TFrame')
log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

ttk.Label(log_frame, text="Activity Log", font=('Helvetica', 12, 'bold')).pack(anchor=tk.W)

activity_text = scrolledtext.ScrolledText(log_frame, height=12, state='disabled', 
                                        font=('Courier', 10), wrap=tk.WORD,
                                        bg='white', fg=text_color, insertbackground=text_color)
activity_text.pack(fill=tk.BOTH, expand=True)

# Update user label when logged in
def update_user_label():
    if current_user:
        user_label.config(text=current_user)
    else:
        user_label.config(text="")
    root.after(100, update_user_label)

update_user_label()

root.mainloop()