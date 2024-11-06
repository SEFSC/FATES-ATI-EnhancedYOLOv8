import cv2
import numpy as np
from ultralytics import YOLO
import torch

# Initialize the YOLO model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('/work/cshah/updatedYOLOv8/ultralytics/runs/detect/train176/weights/best.pt')
model.model.to(device)

# Define the source video and output video
source_video = "/work/cshah/unseen_videos_jackonedrive/2022 Pisces Video/762201029_cam3.avi"
output_video = "output_counting.avi"

# Initialize VideoCapture and VideoWriter
cap = cv2.VideoCapture(source_video)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# Define the counting function
def start_counting(frame, boxes):
    count = len(boxes)
    for box in boxes:
        x1, y1, x2, y2 = box  # Adjusted to handle 4 values
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    
    # Overlay count information
    cv2.putText(frame, f"Count: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    return frame

# Process video frames
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform detection on the frame
    results = model(frame)
    
    # Extract boxes from detection results
    if results and results[0].boxes:
        boxes = results[0].boxes.xyxy.tolist()  # Get bounding boxes in xyxy format
    else:
        boxes = []

    # Apply counting logic
    frame = start_counting(frame, boxes)

    # Write the frame to the output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
