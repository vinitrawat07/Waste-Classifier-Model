import cv2
import numpy as np
import tensorflow as tf

# ── LOAD MODEL ────────────────────────────────────
print("Loading model...")
model = tf.keras.models.load_model("waste_model.h5")
print("Model loaded!")

# ── CONFIG ────────────────────────────────────────
CLASSES  = ["biodegradable", "hazardous", "recyclable"]
IMG_SIZE = (224, 224)

COLORS = {
    "biodegradable": (50,  200, 50),   # green
    "recyclable":    (200, 150, 50),   # blue
    "hazardous":     (50,  50,  220),  # red
}

BINS = {
    "biodegradable": "GREEN BIN",
    "recyclable":    "BLUE BIN",
    "hazardous":     "RED BIN - Handle carefully!",
}

TIPS = {
    "biodegradable": "Food waste, organic matter",
    "recyclable":    "Paper, plastic, cardboard",
    "hazardous":     "Battery, glass, e-waste",
}

def preprocess(frame):
    img = cv2.resize(frame, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32")
    return np.expand_dims(img, 0)

def draw_overlay(frame, label, confidence, color):
    h, w = frame.shape[:2]

    # Draw scanning box in center
    cx, cy = w // 2, h // 2
    size = 150
    cv2.rectangle(frame, (cx-size, cy-size), (cx+size, cy+size), color, 2)
    cv2.putText(frame, "Place waste here", (cx-80, cy-size-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Bottom overlay bar
    cv2.rectangle(frame, (0, h-100), (w, h), (20, 20, 20), -1)

    # Class label
    cv2.putText(frame, label.upper(), (20, h-65),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

    # Bin instruction
    cv2.putText(frame, BINS[label], (20, h-38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Tip
    cv2.putText(frame, TIPS[label], (20, h-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    # Confidence percentage
    cv2.putText(frame, f"{confidence:.1%}", (w-100, h-65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Confidence bar
    bar_w = int((w - 40) * confidence)
    cv2.rectangle(frame, (20, h-8), (w-20, h-3), (60, 60, 60), -1)
    cv2.rectangle(frame, (20, h-8), (20+bar_w, h-3), color, -1)

    return frame

# ── MAIN LOOP ─────────────────────────────────────
print("Starting camera... Press Q to quit")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

label      = "biodegradable"
confidence = 0.0
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not found!")
        break

    # Classify every 15 frames
    if frame_count % 15 == 0:
        try:
            preds      = model.predict(preprocess(frame), verbose=0)[0]
            class_idx  = np.argmax(preds)
            label      = CLASSES[class_idx]
            confidence = float(preds[class_idx])
        except Exception as e:
            print(f"Error: {e}")

    color = COLORS[label]
    frame = draw_overlay(frame, label, confidence, color)

    cv2.imshow("WasteScan AI - Waste Classifier", frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Camera closed.")

