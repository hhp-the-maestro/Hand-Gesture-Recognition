import cv2
import mediapipe as mp
import time
import hand_tracking as ht

capture = cv2.VideoCapture(0)
hand_detection = ht.HandDetector()
p_time = c_time = 0
while True:
    _, img = capture.read()
    img = hand_detection.find_hands(img)
    lm_list = hand_detection.find_position(img, 0, False)

    if len(lm_list) != 0:
        print(lm_list[4])

    c_time = time.time()
    fps = int(1/(c_time-p_time))
    p_time = c_time

    cv2.putText(img, str(fps), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 3, 3)

    cv2.imshow('hand detector', img)

    if cv2.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv2.destroyAllWindows()
