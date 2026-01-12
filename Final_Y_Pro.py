import os
import cv2
import time
import json
import sqlite3
import threading
import numpy as np
import pandas as pd
import pyttsx3
from datetime import datetime, date

from tkinter import *
from tkinter import ttk, messagebox, filedialog

from deepface import DeepFace


DB_PATH = "attendance.db"
IMAGE_DIR = "faces"
EMB_PATH = "embeddings.json"  # cached averaged embeddings per user_id
os.makedirs(IMAGE_DIR, exist_ok=True)


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        roll_no TEXT UNIQUE NOT NULL,
        email TEXT
    )"""
    )
    cur.execute(
        """CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        date TEXT NOT NULL,
        time TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )"""
    )
    # migrate: add email column if missing
    try:
        cur.execute("PRAGMA table_info(users)")
        cols = [r[1] for r in cur.fetchall()]
        if "email" not in cols:
            cur.execute("ALTER TABLE users ADD COLUMN email TEXT")
    except Exception:
        pass
    conn.commit()
    conn.close()


class FaceAttendanceApp:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title("Face Recognition Attendance System")
        self.root.geometry("900x600")

        self.settings = self._load_settings()

        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty("rate", self.settings.get("tts_rate", 170))

        self.cap = None
        self.is_capturing = False
        self.manual_mark_trigger = False
        self.cached_embeddings = self._load_cached_embeddings()

        self._build_ui()

    def _build_ui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=BOTH, expand=True)

        self.frame_register = Frame(notebook)
        self.frame_attend = Frame(notebook)
        self.frame_admin = Frame(notebook)
        self.frame_settings = Frame(notebook)

        notebook.add(self.frame_register, text="Register User")
        notebook.add(self.frame_attend, text="Take Attendance")
        notebook.add(self.frame_admin, text="Admin Panel")
        notebook.add(self.frame_settings, text="Settings")

        self._build_register_ui()
        self._build_attendance_ui()
        self._build_admin_ui()
        self._build_settings_ui()

    # ---------------- REGISTER TAB -----------------
    def _build_register_ui(self):
        Label(self.frame_register, text="Name").pack(pady=5)
        self.entry_name = Entry(self.frame_register, width=30)
        self.entry_name.pack(pady=5)

        Label(self.frame_register, text="Roll No").pack(pady=5)
        self.entry_roll = Entry(self.frame_register, width=30)
        self.entry_roll.pack(pady=5)

        Label(self.frame_register, text="Email (optional)").pack(pady=5)
        self.entry_email = Entry(self.frame_register, width=30)
        self.entry_email.pack(pady=5)

        btn_frame = Frame(self.frame_register)
        btn_frame.pack(pady=10)
        


        self.btn_register = Button(
            btn_frame,
            text="Capture 30 Images (10s)",
            command=self.start_capture_thread,
            bg="#2196F3",
            fg="white",
            padx=10,
            pady=5
        )
        self.btn_register.pack(side=LEFT, padx=5)

        self.lbl_status = Label(self.frame_register, text="Status: Idle", font=("Arial", 10))
        self.lbl_status.pack(pady=10)

    def reset_register_form(self):
        self.entry_name.delete(0, END)
        self.entry_roll.delete(0, END)
        if hasattr(self, "entry_email"):
            self.entry_email.delete(0, END)
        self.lbl_status.config(text="Status: Ready to add new user")

    def start_capture_thread(self):
        if self.is_capturing:
            return
        name = self.entry_name.get().strip()
        roll = self.entry_roll.get().strip()
        email = self.entry_email.get().strip() if hasattr(self, "entry_email") else ""
        if not name or not roll:
            messagebox.showerror("Error", "Please enter name and roll number")
            return
        t = threading.Thread(target=self.capture_images, args=(name, roll, email), daemon=True)
        t.start()

    def capture_images(self, name: str, roll_no: str, email: str = ""):
        self.is_capturing = True
        self.btn_register.config(state="disabled")
        self.lbl_status.config(text="Status: Opening camera...")
        self.root.update_idletasks()

        # Prepare for duplicate check
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT id, name FROM users")
        user_names = {str(r[0]): r[1] for r in cur.fetchall()}
        conn.close()

        known_embeddings = {}
        for uid, emb in self.cached_embeddings.items():
            if uid in user_names:
                known_embeddings[uid] = np.array(emb)

        user_folder = os.path.join(IMAGE_DIR, roll_no)
        os.makedirs(user_folder, exist_ok=True)

        self.cap = self._open_camera()
        if self.cap is None or not self.cap.isOpened():
            self.lbl_status.config(text="Status: Camera error - Check camera permissions")
            self.is_capturing = False
            self.btn_register.config(state="normal")
            messagebox.showerror("Camera Error", 
                "Cannot open camera. Please:\n"
                "1. Check Windows camera permissions\n"
                "2. Close other apps using camera\n"
                "3. Try restarting the application")
            return

        self.lbl_status.config(text="Status: Camera opened - Position your face")
        self.root.update_idletasks()
        
        # Small delay to let camera initialize
        time.sleep(0.5)

        count = 0
        start = time.time()
        window_name = "Registration - Press Q to cancel"
        
        warning_msg = ""
        last_check_time = 0

        try:
            while count < 30 and (time.time() - start) <= 20:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                # Show preview with countdown
                elapsed = time.time() - start
                remaining = max(0, 20 - elapsed)
                countdown_text = f"Capturing: {count}/30 | Time: {remaining:.1f}s"
                
                # Draw text on frame
                cv2.putText(frame, countdown_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Name: {name}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Duplicate Check every 1 second
                if time.time() - last_check_time > 1.0:
                    last_check_time = time.time()
                    try:
                        # Quick check
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        rep = DeepFace.represent(rgb, model_name="SFace", detector_backend="opencv", enforce_detection=False)
                        if rep:
                            curr_emb = rep[0]["embedding"]
                            best_score = -1
                            best_name = ""
                            for uid, k_emb in known_embeddings.items():
                                score = np.dot(curr_emb, k_emb) / (np.linalg.norm(curr_emb) * np.linalg.norm(k_emb))
                                if score > best_score:
                                    best_score = score
                                    best_name = user_names.get(uid, "Unknown")
                            
                            if best_score > float(self.settings.get("threshold", 0.35)):
                                warning_msg = f"WARNING: Already registered as {best_name}"
                                # STRICT: Announce and Stop
                                self._announce(f"User name already exists as {best_name}")
                                self.lbl_status.config(text=f"Status: Failed - Duplicate of {best_name}")
                                # Use root.after to show messagebox on main thread to avoid freezing logic if needed, 
                                # but here we want to stop. 
                                self.root.after(100, lambda: messagebox.showerror("Duplicate Found", f"This face is already registered as {best_name}."))
                                count = -1 
                                break 
                            else:
                                warning_msg = ""
                    except Exception:
                        pass

                if warning_msg:
                    cv2.putText(frame, warning_msg, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Try to detect face for feedback
                try:
                    faces = DeepFace.extract_faces(frame, detector_backend="opencv", enforce_detection=False)
                    if faces:
                        cv2.putText(frame, "Face Detected!", (10, 90), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "No face detected", (10, 90), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                except Exception:
                    pass
                
                # Show preview window
                cv2.imshow(window_name, frame)
                
                # Save image every ~0.66 seconds (30 images in 20 seconds)
                if elapsed >= (count * 20.0 / 30):
                    img_path = os.path.join(user_folder, f"{roll_no}_{count+1}.jpg")
                    cv2.imwrite(img_path, frame)
                    count += 1
                    self.lbl_status.config(text=f"Status: Captured {count}/30 images")
                    self.root.update_idletasks()
                
                # Check for Q key to cancel
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.lbl_status.config(text="Status: Cancelled by user")
                    break
                    
        except Exception as e:
            self.lbl_status.config(text=f"Status: Error - {str(e)}")
            messagebox.showerror("Error", f"Capture error: {str(e)}")
        finally:
            cv2.destroyWindow(window_name)
            if self.cap is not None:
                self.cap.release()
            self.cap = None
            self.is_capturing = False
            self.btn_register.config(state="normal")

        if count == 30:
            self.save_user_to_db(name, roll_no, email)
            self.lbl_status.config(text="Status: Capture complete - Processing...")
            self.root.update_idletasks()
            messagebox.showinfo("Success", f"Registered {name} ({roll_no})\n30 images captured successfully!")
        elif count > 0:
            self.lbl_status.config(text=f"Status: Capture incomplete ({count}/30)")
            messagebox.showwarning("Incomplete", f"Only captured {count}/30 images. Registration may be less accurate.")
        else:
            self.lbl_status.config(text="Status: No images captured")

    def save_user_to_db(self, name: str, roll_no: str, email: str = ""):
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        try:
            cur.execute("INSERT INTO users(name, roll_no, email) VALUES(?, ?, ?)", (name, roll_no, email or None))
            conn.commit()
            user_id = cur.lastrowid
        except sqlite3.IntegrityError:
            messagebox.showwarning("Warning", "Roll number already exists. Updating name and email.")
            cur.execute("UPDATE users SET name=?, email=? WHERE roll_no=?", (name, email or None, roll_no))
            conn.commit()
            cur.execute("SELECT id FROM users WHERE roll_no=?", (roll_no,))
            row = cur.fetchone()
            user_id = row[0] if row else None
        finally:
            conn.close()
        # Build and cache averaged embedding for faster attendance
        if user_id is not None:
            self._build_and_cache_user_embedding(user_id, roll_no)

    # ---------------- ATTENDANCE TAB -----------------
    def _build_attendance_ui(self):
        Label(self.frame_attend, text="Attendance Date (YYYY-MM-DD):").pack(pady=5)
        self.entry_date = Entry(self.frame_attend)
        self.entry_date.insert(0, date.today().isoformat())
        self.entry_date.pack(pady=5)

        self.btn_start_attendance = Button(
            self.frame_attend,
            text="Open Camera (Automatic Attendance)",
            command=self.start_attendance_thread,
        )
        self.btn_start_attendance.pack(pady=10)

        self.lbl_att_status = Label(self.frame_attend, text="Status: Idle")
        self.lbl_att_status.pack(pady=10)


    def start_attendance_thread(self):
        if self.is_capturing:
            return
        att_date = self.entry_date.get().strip()
        try:
            datetime.strptime(att_date, "%Y-%m-%d")
        except ValueError:
            messagebox.showerror("Error", "Invalid date format. Use YYYY-MM-DD")
            return
        t = threading.Thread(target=self.take_attendance, args=(att_date,), daemon=True)
        t.start()

    def _announce(self, text: str):
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception:
            pass

    def _open_camera(self):
        # Try multiple backends/indexes for better Windows compatibility
        preferred_indices = [0, 1, 2]  # Added index 2 as fallback
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        
        for idx in preferred_indices:
            for backend in backends:
                try:
                    cap = cv2.VideoCapture(idx, backend)
                    if cap is not None and cap.isOpened():
                        # Test if we can actually read a frame
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None:
                            try:
                                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                                cap.set(cv2.CAP_PROP_FPS, 30)
                            except Exception:
                                pass
                            return cap
                    if cap is not None:
                        cap.release()
                except Exception as e:
                    # Continue trying other combinations
                    continue
        return None

    def take_attendance(self, att_date: str):
        self.is_capturing = True
        self.lbl_att_status.config(text="Status: Camera Running")

        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT id, name, roll_no FROM users")
        users = cur.fetchall()
        conn.close()

        if not users:
            messagebox.showwarning("Warning", "No users registered")
            self.lbl_att_status.config(text="Status: No users")
            self.is_capturing = False
            return

        # Load from cached embeddings; if missing, try to build and update cache
        known_embeddings = {}
        for user_id, name, roll_no in users:
            emb = self.cached_embeddings.get(str(user_id))
            if emb is None:
                self._build_and_cache_user_embedding(user_id, roll_no)
                emb = self.cached_embeddings.get(str(user_id))
            if emb is not None:
                known_embeddings[user_id] = {
                    "name": name,
                    "roll_no": roll_no,
                    "embedding": np.array(emb),
                }

        if not known_embeddings:
            messagebox.showerror("Error", "No embeddings available for users")
            self.lbl_att_status.config(text="Status: No embeddings")
            self.is_capturing = False
            return

        def cosine(u, v):
            u, v = np.array(u), np.array(v)
            return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))

        threshold = float(self.settings.get("threshold", 0.35))
        last_marked = {}
        dedup_seconds = int(self.settings.get("dedup_seconds", 300))

        self.cap = self._open_camera()
        if self.cap is None or not self.cap.isOpened():
            self.lbl_att_status.config(text="Status: Camera error")
            self.is_capturing = False
            return

        last_result_text = ""
        last_result_time = 0
        last_recognition_time = 0
        recognition_interval = 0.5

        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Automatic Recognition
            if time.time() - last_recognition_time > recognition_interval:
                last_recognition_time = time.time()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                try:
                    rep = DeepFace.represent(
                        rgb,
                        model_name="SFace",
                        detector_backend="opencv",
                        enforce_detection=True,
                    )
                    if rep:
                        emb = rep[0]["embedding"]
                        best_user_id = None
                        best_score = -1
                        for uid, info in known_embeddings.items():
                            score = cosine(emb, info["embedding"])
                            if score > best_score:
                                best_score = score
                                best_user_id = uid

                        if best_score >= threshold:
                            info = known_embeddings[best_user_id]
                            now = datetime.now()
                            now_ts = now.timestamp()
                            if (
                                best_user_id not in last_marked
                                or now_ts - last_marked[best_user_id] > dedup_seconds
                            ):
                                conn = sqlite3.connect(DB_PATH)
                                cur = conn.cursor()
                                cur.execute(
                                    "INSERT INTO attendance(user_id, date, time) VALUES(?,?,?)",
                                    (
                                        best_user_id,
                                        att_date,
                                        now.strftime("%H:%M:%S"),
                                    ),
                                )
                                conn.commit()
                                conn.close()
                                last_marked[best_user_id] = now_ts
                                last_result_text = f"Marked: {info['name']} ({best_score:.2f})"
                                last_result_time = time.time()
                                if self.settings.get("tts_enabled", True):
                                    self._announce(f"Attendance marked for {info['name']}")
                            else:
                                last_result_text = f"Already marked: {info['name']}"
                                last_result_time = time.time()
                except Exception as e:
                    pass
                
            # Draw last result
            if time.time() - last_result_time < 3.0:
                 color = (0, 255, 0) if "Marked" in last_result_text else (0, 0, 255)
                 cv2.putText(frame, last_result_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow("Attendance - Press Q to Stop", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cap.release()
        self.cap = None
        cv2.destroyAllWindows()
        self.is_capturing = False
        self.lbl_att_status.config(text="Status: Stopped")

    # ---------------- ADMIN TAB -----------------
    def _build_admin_ui(self):
        top_frame = Frame(self.frame_admin)
        top_frame.pack(fill=X, pady=5)

        Button(top_frame, text="Refresh Users", command=self.load_users).pack(
            side=LEFT, padx=5
        )
        Button(top_frame, text="Delete Selected User", command=self.delete_user).pack(
            side=LEFT, padx=5
        )



        mid_frame = Frame(self.frame_admin)
        mid_frame.pack(fill=BOTH, expand=True, pady=10)

        columns = ("id", "name", "roll")
        self.tree_users = ttk.Treeview(mid_frame, columns=columns, show="headings")
        for c in columns:
            self.tree_users.heading(c, text=c.upper())
        self.tree_users.column("id", width=50)
        self.tree_users.column("name", width=200)
        self.tree_users.column("roll", width=100)
        self.tree_users.pack(fill=BOTH, expand=True, side=LEFT)

        scroll = ttk.Scrollbar(mid_frame, orient=VERTICAL, command=self.tree_users.yview)
        self.tree_users.configure(yscroll=scroll.set)
        scroll.pack(side=RIGHT, fill=Y)

        bottom_frame = Frame(self.frame_admin)
        bottom_frame.pack(fill=X, pady=5)

        Label(bottom_frame, text="Export Date (YYYY-MM-DD):").pack(side=LEFT, padx=5)
        self.entry_export_date = Entry(bottom_frame, width=12)
        self.entry_export_date.insert(0, date.today().isoformat())
        self.entry_export_date.pack(side=LEFT, padx=5)

        Button(
            bottom_frame,
            text="Export Attendance to Excel",
            command=self.export_attendance,
        ).pack(side=LEFT, padx=5)

        self.load_users()

    def load_users(self):
        for row in self.tree_users.get_children():
            self.tree_users.delete(row)
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT id, name, roll_no FROM users ORDER BY id")
        for row in cur.fetchall():
            self.tree_users.insert("", END, values=row)
        conn.close()

    def delete_user(self):
        item = self.tree_users.focus()
        if not item:
            messagebox.showwarning("Warning", "Select a user to delete")
            return
        user_id, name, roll_no = self.tree_users.item(item)["values"]
        if not messagebox.askyesno(
            "Confirm", f"Delete user {name} ({roll_no}) and all data?"
        ):
            return

        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("DELETE FROM attendance WHERE user_id=?", (user_id,))
        cur.execute("DELETE FROM users WHERE id=?", (user_id,))
        conn.commit()
        conn.close()

        folder = os.path.join(IMAGE_DIR, str(roll_no))
        if os.path.isdir(folder):
            for fname in os.listdir(folder):
                try:
                    os.remove(os.path.join(folder, fname))
                except Exception:
                    pass
            try:
                os.rmdir(folder)
            except Exception:
                pass

        self.load_users()
        messagebox.showinfo("Deleted", "User and data deleted")
        # Also remove cached embedding
        if os.path.exists(EMB_PATH):
            try:
                with open(EMB_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # find deleted user's id by roll_no
                conn = sqlite3.connect(DB_PATH)
                cur = conn.cursor()
                cur.execute("SELECT id FROM users WHERE roll_no=?", (roll_no,))
                # user is already deleted, so skip lookup; instead, remove by any leftover key
                # safest: rebuild cache file to only existing ids
                cur.execute("SELECT id FROM users")
                existing_ids = {str(r[0]) for r in cur.fetchall()}
                conn.close()
                new_data = {k: v for k, v in data.items() if k in existing_ids}
                with open(EMB_PATH, "w", encoding="utf-8") as f:
                    json.dump(new_data, f)
                self.cached_embeddings = new_data
            except Exception:
                pass


    def export_attendance(self):
        exp_date = self.entry_export_date.get().strip()
        try:
            datetime.strptime(exp_date, "%Y-%m-%d")
        except ValueError:
            messagebox.showerror("Error", "Invalid date format")
            return

        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            """SELECT u.id, u.name, u.roll_no, a.date, a.time
            FROM attendance a
            JOIN users u ON a.user_id = u.id
            WHERE a.date = ?
            ORDER BY a.time""",
            (exp_date,),
        )
        rows = cur.fetchall()
        conn.close()

        if not rows:
            messagebox.showwarning("Warning", "No attendance records for this date")
            return

        df = pd.DataFrame(
            rows, columns=["UserID", "Name", "RollNo", "Date", "Time"]
        )

        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel Files", "*.xlsx")],
            title="Save Attendance Excel",
        )
        if not file_path:
            return

        try:
            df.to_excel(file_path, index=False)
            messagebox.showinfo("Success", f"Attendance exported to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export: {e}")

    # ---------------- SETTINGS TAB -----------------
    def _build_settings_ui(self):
        frm = Frame(self.frame_settings)
        frm.pack(fill=BOTH, expand=True, padx=10, pady=10)

        # Threshold
        Label(frm, text="Recognition Threshold (0.2 - 0.6)").grid(row=0, column=0, sticky=W, pady=5)
        self.var_threshold = DoubleVar(value=float(self.settings.get("threshold", 0.35)))
        Entry(frm, textvariable=self.var_threshold, width=10).grid(row=0, column=1, sticky=W)

        # Dedup seconds
        Label(frm, text="Dedup Seconds").grid(row=1, column=0, sticky=W, pady=5)
        self.var_dedup = IntVar(value=int(self.settings.get("dedup_seconds", 300)))
        Entry(frm, textvariable=self.var_dedup, width=10).grid(row=1, column=1, sticky=W)

        # TTS enabled
        self.var_tts = BooleanVar(value=bool(self.settings.get("tts_enabled", True)))
        Checkbutton(frm, text="Enable TTS announcements", variable=self.var_tts).grid(row=2, column=0, columnspan=2, sticky=W, pady=5)

        # TTS rate
        Label(frm, text="TTS Rate").grid(row=3, column=0, sticky=W, pady=5)
        self.var_tts_rate = IntVar(value=int(self.settings.get("tts_rate", 170)))
        Entry(frm, textvariable=self.var_tts_rate, width=10).grid(row=3, column=1, sticky=W)

        # SMTP settings
        Label(frm, text="SMTP Email").grid(row=4, column=0, sticky=W, pady=5)
        self.var_smtp_email = StringVar(value=self.settings.get("smtp_email", ""))
        Entry(frm, textvariable=self.var_smtp_email, width=30).grid(row=4, column=1, sticky=W)

        Label(frm, text="SMTP App Password").grid(row=5, column=0, sticky=W, pady=5)
        self.var_smtp_password = StringVar(value=self.settings.get("smtp_password", ""))
        Entry(frm, textvariable=self.var_smtp_password, width=30, show="*").grid(row=5, column=1, sticky=W)

        Label(frm, text="SMTP Server").grid(row=6, column=0, sticky=W, pady=5)
        self.var_smtp_server = StringVar(value=self.settings.get("smtp_server", "smtp.gmail.com"))
        Entry(frm, textvariable=self.var_smtp_server, width=30).grid(row=6, column=1, sticky=W)

        Label(frm, text="SMTP Port").grid(row=7, column=0, sticky=W, pady=5)
        self.var_smtp_port = IntVar(value=int(self.settings.get("smtp_port", 465)))
        Entry(frm, textvariable=self.var_smtp_port, width=10).grid(row=7, column=1, sticky=W)

        Button(frm, text="Save Settings", command=self._save_settings_clicked).grid(row=8, column=0, pady=10)
        Button(frm, text="Email Absentees for Date", command=self._email_absentees_clicked).grid(row=8, column=1, pady=10, sticky=W)

        for i in range(2):
            frm.grid_columnconfigure(i, weight=1)

    def _load_settings(self):
        path = "settings.json"
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "threshold": 0.35,
            "dedup_seconds": 300,
            "tts_enabled": True,
            "tts_rate": 170,
            "smtp_email": "",
            "smtp_password": "",
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 465,
        }

    def _save_settings_clicked(self):
        self.settings = {
            "threshold": float(self.var_threshold.get()),
            "dedup_seconds": int(self.var_dedup.get()),
            "tts_enabled": bool(self.var_tts.get()),
            "tts_rate": int(self.var_tts_rate.get()),
            "smtp_email": self.var_smtp_email.get().strip(),
            "smtp_password": self.var_smtp_password.get().strip(),
            "smtp_server": self.var_smtp_server.get().strip(),
            "smtp_port": int(self.var_smtp_port.get()),
        }
        with open("settings.json", "w", encoding="utf-8") as f:
            json.dump(self.settings, f)
        self.tts_engine.setProperty("rate", self.settings.get("tts_rate", 170))
        messagebox.showinfo("Saved", "Settings updated")

    def _email_absentees_clicked(self):
        # Ask date to email using file dialog-like prompt; reuse export date if present
        exp_date = self.entry_export_date.get().strip() if hasattr(self, 'entry_export_date') else date.today().isoformat()
        if not exp_date:
            exp_date = date.today().isoformat()
        try:
            datetime.strptime(exp_date, "%Y-%m-%d")
        except ValueError:
            messagebox.showerror("Error", "Invalid date format in Admin export date")
            return
        try:
            count = self._send_absentee_emails_for_date(exp_date)
            messagebox.showinfo("Emails Sent", f"Sent {count} absentee emails for {exp_date}")
        except Exception as e:
            messagebox.showerror("Email Error", str(e))

    def _send_absentee_emails_for_date(self, the_date: str) -> int:
        smtp_email = self.settings.get("smtp_email", "")
        smtp_password = self.settings.get("smtp_password", "")
        smtp_server = self.settings.get("smtp_server", "smtp.gmail.com")
        smtp_port = int(self.settings.get("smtp_port", 465))
        if not smtp_email or not smtp_password:
            raise RuntimeError("Configure SMTP email and app password in Settings.")

        # Build student list and present list
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT id, name, roll_no, email FROM users")
        users = cur.fetchall()
        user_map = {u[0]: {"name": u[1], "roll": u[2], "email": u[3] or ""} for u in users}

        cur.execute("SELECT DISTINCT user_id FROM attendance WHERE date=?", (the_date,))
        present_ids = {r[0] for r in cur.fetchall()}
        conn.close()

        absent = [
            {"id": uid, "name": info["name"], "roll": info["roll"]}
            for uid, info in user_map.items()
            if uid not in present_ids
        ]

        if not absent:
            return 0

        import smtplib
        from email.mime.text import MIMEText

        # Prefer stored email; fallback to synthetic from roll if missing
        def recipient_for(uid, roll, email):
            if email and "@" in email:
                return email
            return f"{roll}@example.com"

        server = smtplib.SMTP_SSL(smtp_server, smtp_port)
        server.login(smtp_email, smtp_password)

        sent = 0
        for item in absent:
            to_addr = recipient_for(item["id"], item["roll"], user_map[item["id"]]["email"])
            body = (
                f"Dear {item['name']} ({item['roll']}),\n\n"
                f"You were marked absent on {the_date}. If this is an error, please contact admin.\n\n"
                f"Regards,\nAttendance System"
            )
            msg = MIMEText(body)
            msg['Subject'] = f"Absence Notice - {the_date}"
            msg['From'] = smtp_email
            msg['To'] = to_addr
            try:
                server.sendmail(smtp_email, [to_addr], msg.as_string())
                sent += 1
            except Exception:
                pass
        server.quit()
        return sent

    # --------- Embedding cache helpers ---------
    def _load_cached_embeddings(self):
        if os.path.exists(EMB_PATH):
            try:
                with open(EMB_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cached_embeddings(self):
        try:
            with open(EMB_PATH, "w", encoding="utf-8") as f:
                json.dump(self.cached_embeddings, f)
        except Exception:
            pass

    def _build_and_cache_user_embedding(self, user_id: int, roll_no: str):
        folder = os.path.join(IMAGE_DIR, roll_no)
        if not os.path.isdir(folder):
            return
        embeddings = []
        for fname in os.listdir(folder):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(folder, fname)
            try:
                rep = DeepFace.represent(
                    img_path,
                    model_name="SFace",
                    detector_backend="opencv",
                    enforce_detection=False,
                )
                if rep:
                    embeddings.append(rep[0]["embedding"])
            except Exception:
                continue
        if embeddings:
            avg = np.mean(np.array(embeddings), axis=0).tolist()
            self.cached_embeddings[str(user_id)] = avg
            self._save_cached_embeddings()


def main():
    init_db()
    root = Tk()
    app = FaceAttendanceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
