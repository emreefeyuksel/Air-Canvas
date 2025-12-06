import cv2
import numpy as np
import mediapipe as mp
from collections import deque

#Drawing Settings
draw_color = (255, 100, 50)
brush_thickness = 8

#Screen Parameters (width, height)
SCREEN_W, SCREEN_H = 1920, 1080

#MediaPipe Settings
mp_hands = mp.solutions.hands
#model_complexity=1 is more precise but slower (optimal for full screen)
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8, model_complexity=1)

points = [deque(maxlen=2048)
index = 0

#Cam start
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

#Window settings
window_name = "Digital Blackboard"
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

#Black Canvas
canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)

print("Kara tahta modu başlatıldı. Çıkmak için 'q', temizlemek için 'c' bas.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)

    if result.multi_hand_landmarks:
        landmarks = result.multi_hand_landmarks[0].landmark

        #Coordinate Scaling for screen parameters
        fore_finger = (int(landmarks[8].x * SCREEN_W), int(landmarks[8].y * SCREEN_H))
        thumb = (int(landmarks[4].x * SCREEN_W), int(landmarks[4].y * SCREEN_H))

        # Çimdik hareketi kontrolü (Hassasiyeti ekran boyutuna göre artırdık: < 3000)
        if (thumb[0] - fore_finger[0]) ** 2 + (thumb[1] - fore_finger[1]) ** 2 < 3000:
            points.append(deque(maxlen=2048))
            index += 1
        else:
            points[index].appendleft(fore_finger)

    #Canvas borders
    for i in range(len(points)):
        for j in range(1, len(points[i])):
            if points[i][j - 1] is None or points[i][j] is None:
                continue

            
            cv2.line(canvas, points[i][j - 1], points[i][j], draw_color, brush_thickness)

    #Showing only the canvas
    cv2.imshow(window_name, canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        #Clear Canvas
        points = [deque(maxlen=2048)]
        index = 0
        canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)

cap.release()

cv2.destroyAllWindows()
