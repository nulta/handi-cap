# Webcam_do.py - gestures to command (actually)

from datetime import date, datetime
from math import floor
import cv2
import numpy as np
from torchvision import models, transforms
import torch
from torch import nn
import pyautogui

CFG = {
    "FRAMES_PER_SECONDS": 60,     # Maximum FPS
    "THRESHOLD_CERTAINTY": 0.80,  # 감지하는 데 필요한 확률의 최솟값.
    "GLOBAL_REQUIRED_FRAMES": 2,  # 딱 이 프레임만큼 연속적으로 감지된 시점에 동작한다.

    # "TARGET_CLASSES": 5,
    "TARGET_CLASSES": 4,
}

NAMES = {
    0: "손바닥",
    1: "옆으로",
    2: "스크롤 아래로",
    3: "배경",
    4: "스크롤 위로",
}


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
pyautogui.FAILSAFE = False

# Model
def import_production_model():
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)  # Pretrained
    model = model.to(device)
    model.classifier[1] = nn.Linear(1536, CFG["TARGET_CLASSES"], True)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    return model

model = import_production_model()

transform_cam = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


prediction_buffer_value = -1
prediction_buffer_count = 0

def predict(frame):
    global prediction_buffer_value
    global prediction_buffer_count

    torch_img = transform_cam(frame).unsqueeze(0)
    logit = model(torch_img)
    certainities = nn.functional.softmax(logit, 1)
    prediction = logit.argmax(dim=1, keepdim=True).item()
    # print(certainities)

    certainity = certainities[0][prediction]
    if certainity < CFG["THRESHOLD_CERTAINTY"]:
        # 확실성이 문턱값 미만; 무시한다.
        return (None, 0)

    return (prediction, certainity)


def process_prediction(prediction, certainity):
    global prediction_buffer_value
    global prediction_buffer_count

    # 버퍼를 쓴다
    if prediction_buffer_value == prediction:
        prediction_buffer_count += 1
    else:
        prediction_buffer_count = 1
        prediction_buffer_value = prediction

    # 미감지 및 "배경"은 무시한다.
    if prediction == None or prediction == 3:
        pyautogui.keyUp("alt")
        return

    print(f"감지: {NAMES[prediction]}, 확실성: {str(floor(certainity * 10000) / 100)}%  (x{prediction_buffer_count})")

    # 행동 실행!!
    count = prediction_buffer_count
    if prediction == 0:
        pyautogui.keyUp("alt")
        if count == 5:
            print("> Alt+F4 (OneShot)")
            pyautogui.hotkey("alt", "f4")

    elif prediction == 1:
        if count == 5:
            pyautogui.keyDown("alt")
            pyautogui.hotkey("tab")
        if (count % 5) == 0:
            print("> Alt+Tab (1/5Shot)")
            pyautogui.hotkey("tab")
            
    elif prediction == 2:
        pyautogui.keyUp("alt")
        print("> ScrollDown (Continous)")
        pyautogui.scroll(-100)

    elif prediction == 4:
        pyautogui.keyUp("alt")
        print("> ScrollUp (Continous)")
        pyautogui.scroll(+100)



# Webcam
webcam = cv2.VideoCapture(0)
print(webcam)
if not webcam.isOpened():
    print("Could not open webcam")
    exit()


# Model + Webcam
next_frametime = datetime.now().timestamp()
while webcam.isOpened():
    # Max FPS 처리
    if next_frametime > datetime.now().timestamp():
        continue
    else:
        next_frametime = datetime.now().timestamp() + (1 / CFG["FRAMES_PER_SECONDS"])


    status, frame = webcam.read()

    if status:
        results = predict(frame)
        process_prediction(results[0], results[1])
        cv2.imshow("test", frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()