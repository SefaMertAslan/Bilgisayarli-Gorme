import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. İYİLEŞTİRME: Bulanıklaştırmayı döngüden önce, TÜM kareye SADECE BİR KEZ uygulayın.
        blurred = cv2.GaussianBlur(frame, (41, 41), 30)
        
        # 2. İYİLEŞTİRME: Maskeyi döngüden önce BİR KEZ oluşturun.
        # Bu maske, tüm yüzlerin birleşimini tutacak.
        composite_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
           for face_landmarks in results.multi_face_landmarks:
                
                # 3. İYİLEŞTİRME (Alternatif): Convex Hull kullan
                
                # Tüm landmark noktalarını (x, y) formatında bir numpy dizisine dönüştür
                points = np.array(
                    [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark],
                    dtype=np.int32
                )
                
                # Noktaların dışbükey örtüsünü (convex hull) hesapla
                hull = cv2.convexHull(points)
                
                # 4. İYİLEŞTİRME: Bulunan yüz şeklini ana maskeye çizin
                cv2.drawContours(composite_mask, [hull], 0, 255, -1)

        # 5. İYİLEŞTİRME: np.where'i döngüden sonra SADECE BİR KEZ çağırın.
        # Maskenin 255 olduğu yerlere bulanık görüntüyü, diğer yerlere orijinal görüntüyü koyun.
        frame = np.where(composite_mask[:, :, None] == 255, blurred, frame)

        cv2.imshow("Blurred Face", frame)

        # ESC (27) ile çık
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()