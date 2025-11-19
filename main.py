import cv2
import numpy as np


# === CẤU HÌNH ===
VIDEO_PATH = "Thailand_traffic_1.mp4" 
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.4
LINE_POSITION = 0.6 


try:
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    print(f"[INFO] Loaded {len(classes)} classes from coco.names")
except FileNotFoundError:
    print("[ERROR] File coco.names not found!")


colors = np.random.uniform(0, 255, size=(80, 3)) 


cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"[ERROR] Cannot open video file: {VIDEO_PATH}")
    exit()

print("[INFO] Starting video stream...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[INFO] Video ended.")
        break
    
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