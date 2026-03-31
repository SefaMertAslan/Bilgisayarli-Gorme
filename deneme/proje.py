import cv2
import numpy as np
import mediapipe as mp
import os
import urllib.request
import time

# ===============================
# MODEL SETUP
# ===============================
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = "hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded ✅")

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ===============================
# HAND DETECTOR SETTINGS
# ===============================
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# ===============================
# CAMERA SETUP
# ===============================
cap = cv2.VideoCapture(0)

# ===============================
# UNLOCK PATTERN
# ===============================
unlock_pattern = ["right", "left", "up", "down"]
detected_pattern = []
unlocked = False
unlock_time = 0

# ===============================
# HELPER: DIRECTION DETECTION
# ===============================
def detect_direction(prev, curr, threshold=40):
    dx = curr[0] - prev[0]
    dy = curr[1] - prev[1]

    if abs(dx) > abs(dy):
        if dx > threshold:
            return "right"
        elif dx < -threshold:
            return "left"
    else:
        if dy < -threshold:
            return "up"
        elif dy > threshold:
            return "down"
    return None


# ===============================
# MAIN LOOP
# ===============================
prev_center = None

print("Camera started. Use your hand to unlock 🔒")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = detector.detect(mp_image)

    # Auto-lock after 5 seconds
    if unlocked and (time.time() - unlock_time > 5):
        unlocked = False
        detected_pattern = []
        print("Lock reactivated 🔒")

    h, w, _ = frame.shape  # Ekran boyutları

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        x = int(hand[9].x * w)
        y = int(hand[9].y * h)

        # Draw hand center point
        cv2.circle(frame, (x, y), 12, (0, 255, 0), -1)

        if prev_center is not None:
            direction = detect_direction(prev_center, (x, y))
            if direction:
                if len(detected_pattern) == 0 or direction != detected_pattern[-1]:
                    detected_pattern.append(direction)
                    print("Gesture:", direction)

        prev_center = (x, y)

        # Draw detected gesture list (center top)
        text = f"Pattern: {detected_pattern}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(frame, text, (text_x, 90),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

        # Check unlock pattern
        if detected_pattern == unlock_pattern and not unlocked:
            unlocked = True
            unlock_time = time.time()
            print("✅ UNLOCKED!")
        elif len(detected_pattern) > len(unlock_pattern):
            print("❌ Wrong pattern! Resetting...")
            detected_pattern = []

    else:
        # No hand detected -> show text at bottom center
        text = "No hand detected"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)[0]
        text_x = (w - text_size[0]) // 2
        text_y = h - 40  # En alt kısım (40px yukarısında)
        cv2.putText(frame, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (180, 180, 180), 2, cv2.LINE_AA)

    # Status box (center top)
    status_text = "LOCK STATUS: UNLOCKED 🔓" if unlocked else "LOCK STATUS: LOCKED 🔒"
    color = (0, 255, 0) if unlocked else (0, 0, 255)
    status_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_DUPLEX, 1, 2)[0]
    status_x = (w - status_size[0]) // 2

    # Transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (status_x - 20, 10), (status_x + status_size[0] + 20, 65), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    cv2.putText(frame, status_text, (status_x, 55),
                cv2.FONT_HERSHEY_DUPLEX, 1, color, 2, cv2.LINE_AA)

    # Show frame
    cv2.imshow("Gesture Unlock System", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('r'):  # Reset
        detected_pattern = []
        unlocked = False
        print("Pattern reset.")

cap.release()
cv2.destroyAllWindows()
