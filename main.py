import cv2
import numpy as np
from centroidtracker import CentroidTracker # Import tracker đã xong

# === CẤU HÌNH ===
VIDEO_PATH = "Thailand_traffic_1.mp4" 
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.4
LINE_POSITION = 0.6 


# Load tên các class COCO
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Các class mục tiêu: bicycle, car, motorbike, bus, truck
TARGET_CLASSES = [1, 2, 3, 5, 7] 

# --- PHẦN CỦA NGƯỜI 2: LOAD MODEL ---
print("[INFO] Loading YOLO from disk...")
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA) # Bật nếu có GPU
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Khởi tạo Tracker (Người 3 đã làm xong class này)
ct = CentroidTracker(maxDisappeared=40)

cap = cv2.VideoCapture(VIDEO_PATH)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (960, 540))
    height, width, channels = frame.shape
    
    # --- PHẦN CỦA NGƯỜI 2: DETECTION ---
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > CONFIDENCE_THRESHOLD and class_id in TARGET_CLASSES:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Resize cho nhẹ
    frame = cv2.resize(frame, (960, 540)) 
    height, width, channels = frame.shape

    # Vẽ thử vạch kẻ (Line) để căn chỉnh vị trí trước
    line_y = int(height * LINE_POSITION)
    cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 255), 2)


    # 1. Detection (Thiện)
    # 2. Tracking (Thông + Thịnh)
    # 3. Counting (Thịnh)


    cv2.imshow("Traffic Counter", frame)
    
 
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()