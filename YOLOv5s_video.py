import cv2
import torch
from PIL import Image

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Videos
video_path = './videos/dashcam.mp4'  
cap = cv2.VideoCapture(video_path)

frame_rate = 0.2 # 每秒捕捉幀數  
prev = 0

while cap.isOpened():
    time_elapsed = 1 / frame_rate
    ret, frame = cap.read()

    if not ret:
        break

    
    current = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

    
    if current - prev > time_elapsed:
        prev = current
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
       
        pil_image = Image.fromarray(rgb_frame)

        
        results = model(pil_image, size=640)

        
        results.print()
        results.save()  

cap.release()