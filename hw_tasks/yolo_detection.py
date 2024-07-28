import torch
import cv2


model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

cap = cv2.VideoCapture('../cv/video_source/road_sample2.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        results = model(frame)
        frame_with_boxes = results.render()[0]

        cv2.imshow('YOLO detection', frame_with_boxes)
        cv2.waitKey(1)
    else:
        break
