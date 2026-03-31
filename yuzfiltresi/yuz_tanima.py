import cv2
import mediapipe as mp
import numpy as np

COLOR = (0, 0, 255)
mp_face_detection = mp.solutions.face_detection

def process_frame(image, draw_box=True):
    """
    Bu fonksiyon, bir görüntüde yüz algılaması yapar ve yüzlerin etrafına sınırlayıcı kutular çizer.
    
    Parametreler:
        - image: İşlenecek görüntü (BGR formatında bir OpenCV çerçevesi).
        - draw_box (bool, opsiyonel): Yüzlerin etrafına sınırlayıcı kutular çizilip çizilmeyeceğini belirler.
          Varsayılan olarak True (kutular çizilir).
    
    Dönüş Değerleri:
        - annotated_image: Yüzlerin etrafına kutular çizilmiş (veya çizilmemiş) işlenmiş görüntü.
        - coordinates: Algılanan her yüzün (origin_x, origin_y, bbox_width, bbox_height) formatında
          koordinatlarını içeren bir liste.
    """
    
    # MediaPipe Yüz Algılama nesnesini başlat
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    # Çerçevenin kopyasını oluştur (orijinal çerçeveyi değiştirmemek için)
    annotated_image = image.copy()
    height, width, _ = image.shape  # Çerçevenin yüksekliğini, genişliğini ve kanal sayısını al
    coordinates = []  # Yüz koordinatlarını saklamak için liste

    # Yüz algılama işlemini gerçekleştir (RGB formatında)
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Yüzler algılandıysa kontrol et
    if results.detections:
        for detection in results.detections:
            # sınırlayıcı kutunun (bounding box) bilgilerini al
            bboxC = detection.location_data.relative_bounding_box
            origin_x = int(bboxC.xmin * width)  # Sol üst köşe x koordinatı
            origin_y = int(bboxC.ymin * height)  # Sol üst köşe y koordinatı
            bbox_width = int(bboxC.width * width)  # Kutunun genişliği
            bbox_height = int(bboxC.height * height)  # Kutunun yüksekliği

            # Koordinatları sınırların içinde kalacak şekilde ayarla
            origin_x = max(0, min(origin_x, width - 1))
            origin_y = max(0, min(origin_y, height - 1))
            bbox_width = max(0, min(bbox_width, width - origin_x))
            bbox_height = max(0, min(bbox_height, height - origin_y))

            # Koordinatları listeye ekle
            coordinates.append((origin_x, origin_y, bbox_width, bbox_height))

            # Eğer draw_box True ise sınırlayıcı kutuyu çiz
            if draw_box:
                start_point = (origin_x, origin_y)  # sınırlayıcı kutunun başlangıç noktası
                end_point = (origin_x + bbox_width, origin_y + bbox_height)  # sınırlayıcı kutunun bitiş noktası
                cv2.rectangle(annotated_image, start_point, end_point, COLOR, 3)  # Mavi kutu çiz

    # İşlenmiş çerçeveyi ve koordinatları döndür
    return annotated_image, coordinates



cap = cv2.VideoCapture(0)
current_filter = None  # aktif filtreyi tutacak

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Hata: Çerçeve yakalanamadı.")
        break
    frame = cv2.flip(frame, 1)


    annotated_frame, face_coordinates = process_frame(frame, draw_box=True)

    key = cv2.waitKey(1) & 0xFF

    # Tuşlara bir kez basıldığında filtre değiştir
    if key in [ord(str(i)) for i in range(1, 8)]:
        current_filter = key
    elif key == ord('0'):
        current_filter = None  # filtreyi kapat
    elif key == ord('q'):
        break

    for (x, y, w, h) in face_coordinates:
        roi = annotated_frame[y:y+h, x:x+w]

        if current_filter == ord('1'):
            filtered = cv2.blur(roi, (15, 15))
        elif current_filter == ord('2'):
            filtered = cv2.medianBlur(roi, 15)
        elif current_filter == ord('3'):
            filtered = cv2.GaussianBlur(roi, (15, 15), 0)
        elif current_filter == ord('4'):
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
            sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
            sobel = cv2.magnitude(sx, sy)
            filtered = cv2.convertScaleAbs(sobel)
            filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
        elif current_filter == ord('5'):
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            kx = np.array([[1, 0, -1],
                           [1, 0, -1],
                           [1, 0, -1]])
            ky = np.array([[1, 1, 1],
                           [0, 0, 0],
                           [-1, -1, -1]])
            px = cv2.filter2D(gray, -1, kx)
            py = cv2.filter2D(gray, -1, ky)
            prewitt = cv2.convertScaleAbs(px + py)
            filtered = cv2.cvtColor(prewitt, cv2.COLOR_GRAY2BGR)
        elif current_filter == ord('6'):
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            filtered = cv2.convertScaleAbs(lap)
            filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
        elif current_filter == ord('7'):
            filtered = cv2.GaussianBlur(roi, (99, 99), 30)
        else:
            filtered = roi

        annotated_frame[y:y+h, x:x+w] = filtered

   
    if current_filter:
        cv2.putText(annotated_frame, f"Filtre: {chr(current_filter)}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Yuz Filtresi", annotated_frame)

cap.release()
cv2.destroyAllWindows()
