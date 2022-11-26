# Webcam.py - For gathering train data
# [c]: save webcam image to file

from datetime import date, datetime
from math import floor
import cv2
import numpy as np

webcam = cv2.VideoCapture(0)
print(webcam)
if not webcam.isOpened():
    print("Could not open webcam")
    exit()

captureTime = 0

while webcam.isOpened():
    status, frame = webcam.read()

    if status:
        img = frame
        if captureTime + 0.08 > datetime.now().timestamp():
            white = np.zeros_like(frame, dtype=np.uint8)
            white.fill(255)
            img = cv2.addWeighted(frame, 1, white, 0.1, 1)
        cv2.imshow("test", img)
    if cv2.waitKey(1) == ord('c'):
        captured = cv2.imwrite(f'capture_{floor(datetime.now().timestamp())}.png', frame)
        captureTime = datetime.now().timestamp()
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()