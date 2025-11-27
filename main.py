import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# --- AYARLAR ---
# Çizim rengi (B, G, R) - Beyaz tebeşir etkisi için (255,255,255) yapabilirsin.
# Şimdilik neon mavi kalsın.
draw_color = (255, 100, 50)
brush_thickness = 8

# --- HEDEF EKRAN ÇÖZÜNÜRLÜĞÜ ---
# Burayı kendi monitör çözünürlüğüne göre (örn: 1920, 1080) ayarlarsan
# çizim hassasiyeti daha iyi olur. Standart olarak bunu kullanıyoruz.
SCREEN_W, SCREEN_H = 1920, 1080

# MediaPipe Ayarları
mp_hands = mp.solutions.hands
# model_complexity=1 daha hassas ama biraz daha yavaştır. Tam ekranda 1 iyidir.
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8, model_complexity=1)

points = [deque(maxlen=2048)]  # Tam ekran için bellek boyutunu artırdık
index = 0

# Kamerayı başlat
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# --- PENCERE AYARLARI (TAM EKRAN İÇİN) ---
window_name = "Digital Blackboard"
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Siyah devasa bir tuval oluştur (Hedef çözünürlükte)
canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)

print("Kara tahta modu başlatıldı. Çıkmak için 'q', temizlemek için 'c' bas.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    # Not: frame.shape'ten gelen h, w'yi artık çizim için KULLANMIYORUZ.

    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)

    if result.multi_hand_landmarks:
        landmarks = result.multi_hand_landmarks[0].landmark

        # --- KRİTİK DEĞİŞİKLİK: KOORDİNAT ÖLÇEKLEME ---
        # AI'dan gelen 0.0-1.0 arasındaki konumu kamera boyutuna göre değil,
        # belirlediğimiz EKRAN boyutuna (SCREEN_W, SCREEN_H) göre çarpıyoruz.
        fore_finger = (int(landmarks[8].x * SCREEN_W), int(landmarks[8].y * SCREEN_H))
        thumb = (int(landmarks[4].x * SCREEN_W), int(landmarks[4].y * SCREEN_H))

        # Çimdik hareketi kontrolü (Hassasiyeti ekran boyutuna göre artırdık: < 3000)
        if (thumb[0] - fore_finger[0]) ** 2 + (thumb[1] - fore_finger[1]) ** 2 < 3000:
            points.append(deque(maxlen=2048))
            index += 1
        else:
            points[index].appendleft(fore_finger)
            # Parmağın ucunu göstermek için tuvale geçici bir daire çizmiyoruz,
            # çünkü bu bir kara tahta deneyimi.

    # Sadece Canvas üzerine çizim yap
    for i in range(len(points)):
        for j in range(1, len(points[i])):
            if points[i][j - 1] is None or points[i][j] is None:
                continue

            # Sadece 'canvas'a çiziyoruz. 'frame' ile işimiz yok.
            cv2.line(canvas, points[i][j - 1], points[i][j], draw_color, brush_thickness)

    # Ekrana sadece siyah tuvali yansıt
    cv2.imshow(window_name, canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        # Tuvali temizle
        points = [deque(maxlen=2048)]
        index = 0
        canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)

cap.release()
cv2.destroyAllWindows()