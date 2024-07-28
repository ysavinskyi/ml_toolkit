import onnxruntime as rt
import numpy as np
import cv2
import ast


inference = rt.InferenceSession('../models/saved_models/ssd_mobilenet_v1_10.onnx')
outputs = ["num_detections:0", "detection_boxes:0", "detection_scores:0", "detection_classes:0"]

filename = '../models/saved_models/coco_classes_map.txt'
coco_classes = {}
with open(filename, 'r') as file:
    for index, line in enumerate(file):
        coco_classes[index] = line.strip()

# cap = cv2.VideoCapture('../cv/video_source/road_sample2.mp4')
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        frame_height, frame_width = frame.shape[:2]
        frame_array = np.expand_dims(frame.astype(np.uint8), axis=0)

        result = inference.run(outputs, {"image_tensor:0": frame_array})
        num_detections, detection_boxes, detection_scores, detection_classes = result

        for i, box in enumerate(detection_boxes[0]):
            y_min, x_min, y_max, x_max = box
            x_min = int(x_min * frame_width)
            y_min = int(y_min * frame_height)
            x_max = int(x_max * frame_width)
            y_max = int(y_max * frame_height)

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            label = int(detection_classes[0][i])
            confidence = detection_scores[0][i]
            obj_class = coco_classes[label] if label <= len(coco_classes) else 'Unknown'

            text = f'Class {obj_class}: {confidence:.2f}%'
            cv2.putText(frame, text, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=0.5,
                        color=(0, 255, 0),
                        thickness=1)

        cv2.imshow('SSD MobileNet detection', frame)
        cv2.waitKey(24)
    else:
        break
