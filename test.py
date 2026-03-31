# Gerekli kütüphanelerin içe aktarılması
# OpenCV: Görüntü işleme ve kamera akışı
# Numpy: Sayısal işlemler
# MediaPipe: El tespiti için makine öğrenimi kütüphanesi
import os  # Dosya işlemleri için
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import urllib.request  # Dosya indirme için

# Model dosyasının URL'si ve yerel dosya yolu
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = "hand_landmarker.task"

# Model dosyasının var olup olmadığını kontrol et ve yoksa indir
def download_model_if_not_exists():
    if not os.path.exists(MODEL_PATH):
        print(f"{MODEL_PATH} bulunamadı. İndiriliyor...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model başarıyla indirildi.")
    else:
        print(f"{MODEL_PATH} zaten mevcut.")

# Görselleştirme ve koordinat döndüren fonksiyon
def draw_landmarks_on_image(rgb_image, detection_result):
    """
    Görüntü üzerine el landmarklarını çizer ve piksel koordinatlarını döndürür.

    Parametreler:
    - rgb_image: RGB formatında giriş görüntüsü
    - detection_result: MediaPipe'in el tespiti sonucu

    Dönen Değerler:
    - annotated_image: İşlenmiş ve çizim yapılmış görüntü
    - all_coordinates: Her elin landmark koordinatları (piksel cinsinden)
    """
    hand_landmarks_list = detection_result.hand_landmarks  # Tespit edilen ellerin listesi
    annotated_image = np.copy(rgb_image)  # Görüntünün kopyası
    height, width, _ = rgb_image.shape  # Görüntünün boyutları

    all_coordinates = []  # Tüm koordinatları saklamak için liste

    # Her tespit edilen el için landmarkları işleme
    for hand_landmarks in hand_landmarks_list:
        # Landmarkları MediaPipe formatına dönüştürme
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
            for landmark in hand_landmarks
        ])

        # Landmarkları görüntü üzerine çizme
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            mp.solutions.hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style()
        )

        # Normalleştirilmiş koordinatları piksel değerine dönüştürme
        coordinates = [
            (int(lm.x * width), int(lm.y * height), lm.z) for lm in hand_landmarks
        ]
        all_coordinates.append(coordinates)  # Her el için koordinatları ekle

    return annotated_image, all_coordinates  # İşlenmiş görüntü ve koordinatları döndür

# Model dosyasını kontrol edip gerekirse indir
download_model_if_not_exists()

# MediaPipe modelini yükleme
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')  # Model dosyasının yolu
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)  # Maksimum 2 el tespiti
detector = vision.HandLandmarker.create_from_options(options)  # Modeli oluştur

# Web kamerayı başlatma
cap = cv2.VideoCapture(0)  # 0, varsayılan kamerayı kullanır

# Kamera akışı boyunca her kareyi işlemek için döngü
while cap.isOpened():
    success, frame = cap.read()  # Kameradan bir kare oku
    frame = cv2.flip(frame, 1)

    if not success:  # Kamera akışı başarısızsa çık
        print("Kamera akışı alınamadı.")
        break

    # OpenCV BGR formatını RGB'ye dönüştürme
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)  # MediaPipe görüntü formatı

    # El tespiti yap
    detection_result = detector.detect(mp_image)

    # İşlenmiş görüntüyü ve koordinatları al
    annotated_frame, coordinates = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)

    # Koordinatları konsola yazdır
    if coordinates:
        print(f"Piksel cinsinden el koordinatları: {coordinates}")

    # İşlenmiş görüntüyü ekranda göster
    cv2.imshow('El Tespiti', cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

    # ESC tuşuna basıldığında döngüyü sonlandır
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Kamera akışını serbest bırak ve pencereleri kapat
cap.release()
cv2.destroyAllWindows()



"""
Kodun Özet Açıklaması
Kütüphanelerin İçe Aktarılması: OpenCV ile kamera akışı sağlanır, MediaPipe ile el tespiti yapılır.
El Landmarklarının Çizimi:
Her landmark’ın koordinatları normalize edilmiş (x, y, z) olarak hesaplanır.
(x, y) değerleri piksel cinsine çevrilir.
Modelin Yüklenmesi: MediaPipe'in el tespit modeli kullanılarak maksimum iki el tespit edilir.
Kamera Döngüsü:
Her karede el tespiti yapılarak landmarklar çizilir.
Landmark koordinatları konsola yazdırılır.
ESC tuşuna basıldığında döngüden çıkılır.d
"""

