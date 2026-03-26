import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import os
import sqlite3
import threading
from datetime import datetime
from collections import deque, Counter

# ── LOAD MODELS ───────────────────────────────────
print("Loading models...")
waste_model = tf.keras.models.load_model("waste_model.h5")
yolo        = YOLO("yolov8n.pt")
print("Models loaded!")

# ── CONFIG ────────────────────────────────────────
CLASSES        = ["biodegradable", "hazardous", "recyclable"]
IMG_SIZE       = (224, 224)
CONF_THRESHOLD = 0.65
SMOOTH_FRAMES  = 10
RETRAIN_EVERY  = 50

COLORS = {
    "biodegradable": (50,  200, 50),
    "recyclable":    (200, 150, 50),
    "hazardous":     (50,  50,  220),
    "unknown":       (150, 150, 150),
}

BINS = {
    "biodegradable": "GREEN BIN",
    "recyclable":    "BLUE BIN",
    "hazardous":     "RED BIN - Handle carefully!",
    "unknown":       "UNKNOWN - Press B / R / H to teach me!",
}

KEY_MAP = {
    ord('b'): "biodegradable",
    ord('r'): "recyclable",
    ord('h'): "hazardous",
}

# ── CREATE FOLDERS ────────────────────────────────
for c in CLASSES + ["unknown"]:
    os.makedirs(f"corrections/{c}", exist_ok=True)

# ── DATABASE SETUP (Approach B) ───────────────────
def init_db():
    conn = sqlite3.connect("corrections.db")
    c    = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS corrections (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path  TEXT,
            label       TEXT,
            timestamp   TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS overrides (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            image_hash   TEXT UNIQUE,
            label        TEXT,
            timestamp    TEXT
        )
    """)
    conn.commit()
    conn.close()

def get_correction_count():
    conn  = sqlite3.connect("corrections.db")
    c     = conn.cursor()
    count = c.execute("SELECT COUNT(*) FROM corrections").fetchone()[0]
    conn.close()
    return count

def save_correction_db(image_path, label, image_hash):
    conn = sqlite3.connect("corrections.db")
    c    = conn.cursor()
    c.execute("INSERT INTO corrections (image_path, label, timestamp) VALUES (?, ?, ?)",
              (image_path, label, datetime.now().strftime("%Y%m%d_%H%M%S")))
    c.execute("INSERT OR REPLACE INTO overrides (image_hash, label, timestamp) VALUES (?, ?, ?)",
              (image_hash, label, datetime.now().strftime("%Y%m%d_%H%M%S")))
    conn.commit()
    conn.close()

def check_override(image_hash):
    conn   = sqlite3.connect("corrections.db")
    c      = conn.cursor()
    result = c.execute("SELECT label FROM overrides WHERE image_hash = ?",
                       (image_hash,)).fetchone()
    conn.close()
    return result[0] if result else None

def get_image_hash(crop):
    # Simple hash based on resized image pixels
    small = cv2.resize(crop, (16, 16)).flatten()
    return str(hash(small.tobytes()))

init_db()

# ── BACKGROUND RETRAINING (Approach A) ────────────
is_retraining  = False
retrain_status = ""

def retrain_on_corrections():
    global is_retraining, retrain_status, waste_model

    is_retraining  = True
    retrain_status = "Retraining..."
    print("\nAuto retraining started...")

    try:
        conn  = sqlite3.connect("corrections.db")
        c     = conn.cursor()
        rows  = c.execute("SELECT image_path, label FROM corrections").fetchall()
        conn.close()

        if len(rows) == 0:
            retrain_status = "No corrections found"
            is_retraining  = False
            return

        # Build training data from correction images
        images, labels = [], []
        for image_path, label in rows:
            if not os.path.exists(image_path):
                continue
            try:
                img = cv2.imread(image_path)
                img = cv2.resize(img, IMG_SIZE)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32")
                images.append(img)
                labels.append(CLASSES.index(label))
            except:
                continue

        if len(images) == 0:
            retrain_status = "No valid images"
            is_retraining  = False
            return

        X = np.array(images)
        Y = np.array(labels)

        print(f"Retraining on {len(images)} correction images...")
        retrain_status = f"Training on {len(images)} corrections..."

        waste_model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        waste_model.fit(X, Y, epochs=10, batch_size=8, verbose=1)
        waste_model.save("waste_model.h5")

        retrain_status = f"Retrained on {len(images)} corrections!"
        print(f"Retraining complete! Model saved.")

    except Exception as e:
        retrain_status = f"Retrain error: {str(e)}"
        print(f"Retrain error: {e}")

    is_retraining = False

def trigger_retrain_if_needed():
    count = get_correction_count()
    if count > 0 and count % RETRAIN_EVERY == 0 and not is_retraining:
        thread = threading.Thread(target=retrain_on_corrections)
        thread.daemon = True
        thread.start()

# ── PREDICTION SMOOTHER ───────────────────────────
smoothers = [deque(maxlen=SMOOTH_FRAMES) for _ in range(20)]

def smooth_prediction(slot_idx, label, confidence):
    smoothers[slot_idx].append((label, confidence))
    if len(smoothers[slot_idx]) < 3:
        return label, confidence
    labels      = [x[0] for x in smoothers[slot_idx]]
    most_common = Counter(labels).most_common(1)[0][0]
    avg_conf    = np.mean([x[1] for x in smoothers[slot_idx]
                           if x[0] == most_common])
    return most_common, avg_conf

def reset_smoother(slot_idx):
    smoothers[slot_idx].clear()

# ── CLASSIFY CROP ─────────────────────────────────
def classify_crop(crop):
    try:
        # Check override database first (Approach B)
        img_hash = get_image_hash(crop)
        override = check_override(img_hash)
        if override:
            return override, 1.0

        # Run model
        img   = cv2.resize(crop, IMG_SIZE)
        img   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32")
        img   = np.expand_dims(img, 0)
        preds = waste_model.predict(img, verbose=0)[0]
        conf  = float(np.max(preds))
        label = CLASSES[np.argmax(preds)] if conf >= CONF_THRESHOLD else "unknown"
        return label, conf
    except:
        return "unknown", 0.0

# ── SAVE UNKNOWN ──────────────────────────────────
def save_unknown(crop):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path      = f"corrections/unknown/{timestamp}.jpg"
    cv2.imwrite(path, crop)

# ── CORRECTION ────────────────────────────────────
def correct_and_learn(crop, correct_class, slot_idx):
    print(f"\nCorrection: {correct_class.upper()}")

    # Save image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    save_path = f"corrections/{correct_class}/{timestamp}.jpg"
    cv2.imwrite(save_path, crop)

    # Get image hash for override
    img_hash = get_image_hash(crop)

    # Save to database (Approach B — instant override)
    save_correction_db(save_path, correct_class, img_hash)
    count = get_correction_count()
    print(f"Saved to database! Total corrections: {count}")

    # Reset smoother so new label shows immediately
    reset_smoother(slot_idx)

    # Trigger background retrain if needed (Approach A)
    trigger_retrain_if_needed()

    return correct_class

# ── DRAW BOX ──────────────────────────────────────
def draw_box(frame, box, label, confidence, is_selected):
    x1, y1, x2, y2 = box
    color     = COLORS[label]
    box_color = (0, 255, 255) if is_selected else color
    thickness = 3 if is_selected else 2

    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)

    conf_text  = f"{confidence:.0%}" if label != "unknown" else "?"
    label_text = f"{label}  {conf_text}"
    font_scale = 0.52
    (tw, th), _ = cv2.getTextSize(
        label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

    cv2.rectangle(frame, (x1, y1-th-10), (x1+tw+10, y1), box_color, -1)
    text_color = (0, 0, 0)
    cv2.putText(frame, label_text, (x1+5, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)
    cv2.putText(frame, BINS[label], (x1, y2+18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)

    if is_selected:
        cv2.putText(frame, "[ SELECTED ]", (x1, y1-th-14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)

# ── MAIN LOOP ─────────────────────────────────────
print("Starting camera...")
print("Controls: TAB=select   B=biodegradable   R=recyclable   H=hazardous   Q=quit")

cap = cv2.VideoCapture(1)  # or 2 if 1 doesn't work
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

detections     = []
selected_idx   = 0
feedback_msg   = ""
feedback_timer = 0
frame_count    = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not found!")
        break

    h, w = frame.shape[:2]

    # ── DETECT + CLASSIFY every 20 frames ─────────
    if frame_count % 20 == 0:
        new_detections = []
        results = yolo(frame, verbose=False)[0]

        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            pad = 8
            x1  = max(0, x1-pad)
            y1  = max(0, y1-pad)
            x2  = min(w, x2+pad)
            y2  = min(h, y2+pad)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            raw_label, raw_conf   = classify_crop(crop)
            label, confidence     = smooth_prediction(i, raw_label, raw_conf)

            if label == "unknown":
                save_unknown(crop)

            new_detections.append(((x1, y1, x2, y2), label, confidence, crop))

        detections = new_detections
        if selected_idx >= len(detections):
            selected_idx = 0

    # ── DRAW ──────────────────────────────────────
    for i, (box, label, conf, crop) in enumerate(detections):
        draw_box(frame, box, label, conf, is_selected=(i == selected_idx))

    if len(detections) == 0:
        cv2.putText(frame, "No objects detected — show waste to camera",
                    (20, h//2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (150, 150, 150), 1)

    # ── TOP BAR ───────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 28), (20, 20, 20), -1)
    correction_count = get_correction_count()
    next_retrain     = RETRAIN_EVERY - (correction_count % RETRAIN_EVERY)
    status_text      = retrain_status if is_retraining else \
                       f"Corrections: {correction_count}  Next retrain in: {next_retrain}"
    cv2.putText(frame, f"WasteScan AI   Objects: {len(detections)}   {status_text}",
                (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

    # ── RETRAIN INDICATOR ─────────────────────────
    if is_retraining:
        cv2.rectangle(frame, (0, 28), (w, 50), (0, 80, 0), -1)
        cv2.putText(frame, f"RETRAINING IN BACKGROUND: {retrain_status}",
                    (10, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    # ── BOTTOM BAR ────────────────────────────────
    cv2.rectangle(frame, (0, h-24), (w, h), (20, 20, 20), -1)
    cv2.putText(frame,
                "TAB: select    B: biodegradable    R: recyclable    H: hazardous    Q: quit",
                (8, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (150, 150, 150), 1)

    # ── FEEDBACK ──────────────────────────────────
    if feedback_timer > 0:
        cv2.rectangle(frame, (0, 28), (w, 58), (0, 60, 0), -1)
        cv2.putText(frame, feedback_msg, (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        feedback_timer -= 1
    else:
        feedback_msg = ""

    cv2.imshow("WasteScan AI - Multi Object Classifier", frame)
    frame_count += 1

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key == 9 and len(detections) > 0:
        selected_idx = (selected_idx + 1) % len(detections)
        reset_smoother(selected_idx)

    if key in KEY_MAP and len(detections) > 0:
        correct_class             = KEY_MAP[key]
        box, label, conf, crop    = detections[selected_idx]
        correct_and_learn(crop, correct_class, selected_idx)
        detections[selected_idx]  = (box, correct_class, 1.0, crop)
        count                     = get_correction_count()
        next_retrain              = RETRAIN_EVERY - (count % RETRAIN_EVERY)
        feedback_msg              = f"Saved! {correct_class.upper()} | " \
                                    f"Total corrections: {count} | " \
                                    f"Retrain in: {next_retrain} more"
        feedback_timer            = 90

cap.release()
cv2.destroyAllWindows()
print("Camera closed.")