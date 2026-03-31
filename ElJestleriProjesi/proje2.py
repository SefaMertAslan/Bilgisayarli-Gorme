# proje2_full_mirror_landmarks_v5.py
import cv2
import numpy as np
import mediapipe as mp
import math
from collections import deque
import urllib.request
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ------------------------------
# MODEL
# ------------------------------
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = "hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# ------------------------------
# CAMERA
# ------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

# ------------------------------
# HELPERS
# ------------------------------
mp_hands = mp.solutions.hands
HAND_CONNECTIONS = mp_hands.HAND_CONNECTIONS

def rotate(frame, angle):
    h, w = frame.shape[:2]
    M = cv2.getRotationMatrix2D((w//2,h//2), angle, 1)
    return cv2.warpAffine(frame, M, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def adjust_hsv_sv(frame, delta):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    s = np.clip(s.astype(np.int16)+delta,0,255).astype(np.uint8)
    v = np.clip(v.astype(np.int16)+delta,0,255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([h,s,v]), cv2.COLOR_HSV2BGR)

def adjust_contrast(frame, alpha=1.5):
    alpha = max(0.1, min(5.0, alpha))
    return cv2.convertScaleAbs(frame, alpha=alpha)

def is_hand_open(hand):
    try:
        # 8, 12, 16, 20 parmak uçları, 6, 10, 14, 18 orta boğumlar.
        # En az 3 parmak ucu orta boğumdan (y-ekseninde) yukarıdaysa açık.
        return sum(1 for i in [8,12,16,20] if hand[i].y < hand[i-2].y) >= 3
    except:
        return False

def is_hand_closed(hand):
    try:
        # En az 3 parmak ucu orta boğumdan (y-ekseninde) aşağıdaysa kapalı.
        return sum(1 for i in [8,12,16,20] if hand[i].y > hand[i-2].y) >= 3
    except:
        return False

def get_center(hand, w, h):
    # Bilek ile orta parmak tabanı arasındaki nokta (9. landmark)
    return int(hand[9].x*w), int(hand[9].y*h)

def get_distance(h1, h2, w, h):
    return math.hypot((h1[9].x - h2[9].x)*w, (h1[9].y - h2[9].y)*h)

def get_angle(hand, w, h):
    # Bilek (0) ve orta parmak ucu (12) arasındaki açıyı hesapla
    p0 = (hand[0].x*w, hand[0].y*h)
    p12 = (hand[12].x*w, hand[12].y*h)
    dx, dy = p12[0]-p0[0], p12[1]-p0[1]
    # +90 derece ekleyerek dik (dikey) açıyı sıfır referansı yapıyoruz
    return math.degrees(math.atan2(dy,dx))+90

# ---  LANDMARK ÇİZİM FONKSİYONU ---
def draw_hand_landmarks(frame, hand, w, h):
  
    overlay = frame.copy()

    # Landmark noktaları ve gölgeleri
    for i, lm in enumerate(hand):
        x, y = int(lm.x * w), int(lm.y * h)

        # Gölge
        cv2.circle(overlay, (x + 2, y + 2), 7, (50, 50, 50), -1)  # Koyu gri gölge

        # Ana nokta
        if i in [8, 12, 16, 20]:  # Parmak uçları (Tip)
            cv2.circle(overlay, (x, y), 6, (0, 255, 0), -1)  # Yeşil vurgu
            cv2.circle(overlay, (x, y), 9, (255, 255, 255), 1) # Beyaz dış halka
        elif i == 0: # Bilek (Wrist)
            cv2.circle(overlay, (x, y), 8, (255, 0, 0), -1) # Mavi vurgu
        else: # Diğer noktalar
            cv2.circle(overlay, (x, y), 5, (0, 0, 255), -1)  # Kırmızı nokta

    # Transparan overlay'i ana kareye ekle (Sadece noktalar için)
    alpha_dots = 0.8  # Noktalar daha belirgin
    cv2.addWeighted(overlay, alpha_dots, frame, 1 - alpha_dots, 0, frame)

    # Parmak bağlantıları: Turkuaz ve transparan
    for start_idx, end_idx in HAND_CONNECTIONS:
        x0, y0 = int(hand[start_idx].x * w), int(hand[start_idx].y * h)
        x1, y1 = int(hand[end_idx].x * w), int(hand[end_idx].y * h)

        # Çizgiye saydamlık uygulamak için ayrı bir overlay kullanılır.
        overlay_line = frame.copy()
        line_color = (255, 255, 0) # Sarı-Turkuaz
        line_thickness = 4
        cv2.line(overlay_line, (x0, y0), (x1, y1), line_color, line_thickness)

        # Çizgiyi ana kareye saydam olarak ekle
        alpha_lines = 0.4 # Çizgiler daha saydam
        cv2.addWeighted(overlay_line, alpha_lines, frame, 1 - alpha_lines, 0, frame)

# ------------------------------
# STATE
# ------------------------------
prev_left_y = deque(maxlen=5)
brightness = 0
prev_dist = None
mirror_mode = False
mirror_cooldown = 0

print("Camera ready. Perform gestures...")

# ------------------------------
# MAIN LOOP
# ------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame,1)
    h, w, _ = frame.shape
    processed = frame.copy()

    # Prepare MediaPipe image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    hands_list = result.hand_landmarks if result and result.hand_landmarks else []

    effect = "NONE"

    # assign left/right based on center
    left_hand = right_hand = None
    left_center = right_center = None
    for hnd in hands_list:
        cx, cy = get_center(hnd, w, h)
        # Kamera ters çevrildiği için (cv2.flip(frame,1)), soldaki el gerçek hayatta
        # ekranın sağ yarısında görünür, ancak burada merkezine göre ayrım yapıyoruz.
        if cx < w//2 and left_hand is None:
            left_hand = hnd; left_center = (cx, cy)
        elif cx >= w//2 and right_hand is None:
            right_hand = hnd; right_center = (cx, cy)
        # Landmark çizimi
        draw_hand_landmarks(processed, hnd, w, h)

    # --- İKİ EL VAR ---
    if left_hand and right_hand:
        lx, ly = left_center
        rx, ry = right_center

        # ayna modu toggle: iki el kapalı ve aynı hizada
        if is_hand_closed(left_hand) and is_hand_closed(right_hand):
            if abs(ly - ry) < h*0.08:
                if mirror_cooldown == 0:
                    mirror_mode = not mirror_mode
                    mirror_cooldown = 15
                effect = "MIRROR"

        # kontrast kontrolü: iki el açık ve aynı hizada
        if is_hand_open(left_hand) and is_hand_open(right_hand):
            if abs(ly-ry) < h*0.08:
                processed = adjust_contrast(processed, alpha=2.0)
                effect = "CONTRAST"

        # parlaklık mesafe değişimi
        dist = get_distance(left_hand, right_hand, w, h)
        if prev_dist is not None:
            diff = dist - prev_dist
            if diff < -15:
                brightness += 15 # Elleri yaklaştırmak parlaklığı artırır
                effect = "BRIGHT_UP"
            elif diff > 15:
                brightness -= 15 # Elleri uzaklaştırmak parlaklığı azaltır
                effect = "BRIGHT_DOWN"

        brightness = np.clip(brightness, -100, 100) # Parlaklığı sınırla
        tmp = processed.astype(np.int16) + brightness
        processed = np.clip(tmp,0,255).astype(np.uint8)
        prev_dist = dist

    # --- SOL EL TEK ---
    elif left_hand:
        if is_hand_open(left_hand):
            # ROTATE: Sol el açıkken rotasyon
            ang = get_angle(left_hand, w, h)
            processed = rotate(processed, ang)
            effect = "ROTATE"
        if is_hand_closed(left_hand):
            # HSV/SATURATION: Sol el kapalıyken Y hareketi
            cy = left_center[1]
            prev_left_y.append(cy)
            if len(prev_left_y) == prev_left_y.maxlen:
                avg_start = np.mean(list(prev_left_y)[:3])
                avg_end = np.mean(list(prev_left_y)[-3:])
                dy = avg_end - avg_start
                if dy < -5: # Hızlı yukarı hareket (y küçülür)
                    processed = adjust_hsv_sv(processed, +20)
                    effect = "HSV_UP"
                elif dy > 5: # Hızlı aşağı hareket (y büyür)
                    processed = adjust_hsv_sv(processed, -20)
                    effect = "HSV_DOWN"

    # --- SAĞ EL TEK ---
    elif right_hand:
        if is_hand_open(right_hand):
            # NEGATIVE: Sağ el açık
            processed = cv2.bitwise_not(processed)
            effect = "NEGATIVE"
        elif is_hand_closed(right_hand):
            # GRAYSCALE: Sağ el kapalı
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            effect = "GRAYSCALE"
            
    # Eğer tek el varsa, mesafeyi sıfırla
    if not (left_hand and right_hand):
        prev_dist = None

    # mirror uygulama
    if mirror_mode:
        processed = cv2.flip(processed,1)

    if mirror_cooldown > 0:
        mirror_cooldown -= 1

    # Ekrana bilgi
    status_text = f"Effect: {effect}"
    status_text += " | MIRROR ON" if mirror_mode else " | MIRROR OFF"
    cv2.putText(processed, status_text, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(processed, f"Brightness: {brightness}", (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)


    cv2.imshow("Gesture Control", processed)

    key = cv2.waitKey(1)&0xFF
    if key==27:
        break
    

cap.release()
cv2.destroyAllWindows()