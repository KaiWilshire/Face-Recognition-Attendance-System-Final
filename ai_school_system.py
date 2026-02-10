import os
import sys
import cv2
import pickle
import threading
import datetime
import time
import pandas as pd
import numpy as np

# ---------------- TENSORFLOW / DEEPFACE ENV ----------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import customtkinter as ctk
from tkinter import messagebox
from deepface import DeepFace
from scipy.spatial.distance import cosine
import traceback

# ---------------- EXCEPTION VISIBILITY ----------------
def excepthook(type, value, tb):
    traceback.print_exception(type, value, tb)

sys.excepthook = excepthook

# ---------------- PATH HANDLING (EXE SAFE) ----------------
def get_app_path():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

BASE_DIR = get_app_path()

DATA_PATH = os.path.join(BASE_DIR, "face_id_data")
ENCODINGS_FILE = os.path.join(BASE_DIR, "face_encodings.pkl")
ATTENDANCE_FILE = os.path.join(BASE_DIR, "attendance_log.csv")

os.makedirs(DATA_PATH, exist_ok=True)

# ---------------- CONFIG ----------------
MODEL_NAME = "VGG-Face"
THRESHOLD = 0.5
MAX_IMAGES_PER_USER = 50
WATERMARK_TEXT = "Zwe Maung Maung Than"

recent_logs = []
is_recognizing = False
status_message = "Ready"

# ---------------- UTIL ----------------
def draw_watermark(frame):
    h, w, _ = frame.shape
    cv2.putText(
        frame,
        WATERMARK_TEXT,
        (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

# ---------------- REGISTRATION ----------------
def run_registration_logic(name):
    last_capture = 0
    user_dir = os.path.join(DATA_PATH, name)
    os.makedirs(user_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = len(os.listdir(user_dir))

    instructions = ["Look straight", "Look LEFT", "Look RIGHT", "Look UP", "Look DOWN"]

    while cap.isOpened() and count < MAX_IMAGES_PER_USER:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        instr = instructions[(count // 10) % len(instructions)]

        cv2.putText(frame, f"Registering: {name}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Instruction: {instr}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Images: {count}/{MAX_IMAGES_PER_USER}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        draw_watermark(frame)

        if time.time() - last_capture > 0.3:
            cv2.imwrite(os.path.join(user_dir, f"{count}.jpg"), frame)
            count += 1
            last_capture = time.time()

        cv2.imshow("Registration (ESC to stop)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------------- ENCODING ----------------
def sync_db():
    encodings = []
    names = []

    for user in os.listdir(DATA_PATH):
        user_path = os.path.join(DATA_PATH, user)
        if not os.path.isdir(user_path):
            continue

        images = [
            os.path.join(user_path, img)
            for img in os.listdir(user_path)
            if img.lower().endswith((".jpg", ".png"))
        ]

        if not images:
            continue

        print(f"[INFO] Encoding user: {user}")
        success = False

        for img_path in images[:5]:
            try:
                res = DeepFace.represent(
                    img_path,
                    model_name=MODEL_NAME,
                    detector_backend="skip",
                    enforce_detection=False,
                )

                if res and "embedding" in res[0]:
                    encodings.append(res[0]["embedding"])
                    names.append(user)
                    success = True
                    break
            except Exception as e:
                print(f"[WARN] Failed on {img_path}: {e}")

        if not success:
            print(f"[ERROR] No valid face found for {user}")

    if encodings:
        with open(ENCODINGS_FILE, "wb") as f:
            pickle.dump({"encodings": encodings, "names": names}, f)
        print(f"[OK] Encodings saved: {len(encodings)}")
    else:
        print("[FATAL] No encodings generated")

    return encodings, names

# ---------------- ATTENDANCE ----------------
def run_attendance_logic(session_name):
    global is_recognizing, status_message

    known_enc, known_names = sync_db()
    if not known_enc:
        messagebox.showerror("Error", "No encodings available.")
        return

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        draw_watermark(frame)

        if not is_recognizing:
            def recognize(img):
                global is_recognizing, status_message
                is_recognizing = True
                try:
                    res = DeepFace.represent(
                        img,
                        model_name=MODEL_NAME,
                        detector_backend="skip",
                        enforce_detection=False,
                    )
                    emb = res[0]["embedding"]
                    dists = [cosine(emb, k) for k in known_enc]

                    if dists and min(dists) < THRESHOLD:
                        name = known_names[dists.index(min(dists))]
                        now = datetime.datetime.now()

                        log = {
                            "Name": name,
                            "Date": now.strftime("%Y-%m-%d"),
                            "Time": now.strftime("%H:%M:%S"),
                            "Session": session_name,
                        }

                        pd.DataFrame([log]).to_csv(
                            ATTENDANCE_FILE,
                            mode="a",
                            header=not os.path.exists(ATTENDANCE_FILE),
                            index=False,
                        )

                        recent_logs.insert(0, f"{log['Time']} - {name}")
                        status_message = f"Marked: {name}"
                    else:
                        status_message = "Unknown"
                except Exception as e:
                    print("[ERR]", e)
                    status_message = "Scanning..."

                is_recognizing = False

            threading.Thread(target=recognize, args=(frame.copy(),), daemon=True).start()

        cv2.putText(frame, status_message, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Attendance (ESC to close)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------------- UI ----------------
class Dashboard(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI Attendance System")
        self.geometry("900x500")

        sidebar = ctk.CTkFrame(self, width=220)
        sidebar.pack(side="left", fill="y", padx=10, pady=10)

        ctk.CTkLabel(sidebar, text="ADMIN PANEL",
                     font=("Arial", 18, "bold")).pack(pady=20)

        ctk.CTkButton(sidebar, text="Register Student",
                      command=self.register).pack(pady=10)

        ctk.CTkButton(sidebar, text="Start Attendance",
                      fg_color="green",
                      command=self.attendance).pack(pady=10)

        ctk.CTkButton(sidebar, text="View Attendance Log",
                      command=self.open_log).pack(pady=10)

        self.log_box = ctk.CTkTextbox(self)
        self.log_box.pack(expand=True, fill="both", padx=10, pady=10)

        self.update_logs()

    def register(self):
        name = ctk.CTkInputDialog(
            title="Register",
            text="Enter student name:",
        ).get_input()

        if name:
            threading.Thread(
                target=run_registration_logic,
                args=(name.strip(),),
                daemon=True,
            ).start()

    def attendance(self):
        session = ctk.CTkInputDialog(
            title="Session",
            text="Enter class/session name:",
        ).get_input()

        if session:
            threading.Thread(
                target=run_attendance_logic,
                args=(session,),
                daemon=True,
            ).start()

    def open_log(self):
        if os.path.exists(ATTENDANCE_FILE):
            os.startfile(ATTENDANCE_FILE)
        else:
            messagebox.showinfo("No Log", "Attendance log not found yet.")

    def update_logs(self):
        self.log_box.delete("1.0", "end")
        self.log_box.insert("1.0", "\n".join(recent_logs[:20]))
        self.after(2000, self.update_logs)

# ---------------- RUN ----------------
if __name__ == "__main__":
    app = Dashboard()
    app.mainloop()
