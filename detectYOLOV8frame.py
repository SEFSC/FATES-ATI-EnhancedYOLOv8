import cv2
import time
import torch
from pathlib import Path
from ultralytics import YOLO
import numpy as np

### Load YOLOv8 model
#weights_path = '/work/cshah/YOLOv8_weights_saved/YOLOvn/weights/best.pt'

model = YOLO('/work/cshah/YOLOv8_weights_saved/YOLOvn/weights/best.pt')


# Load video
video_path = '../../datasets/2021TestVideo/762101178_cam3.avi'
cap = cv2.VideoCapture(video_path)

### Define output video path and codec
output_path = '/work/cshah/YOLOv8/runs/detect/output_video.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

# Initialize variables for FPS calculation
fps_start_time = time.time()
frame_count = 0

# Helper function to round bounding box coordinates
def round_box(box):
    return [int(round(coord)) for coord in box]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform YOLOv8 detection
    results = model(frame)

    # Access the first element of the results list
    detections = results[0]
    print('detections frame',detections)

    ### Draw bounding boxes on the frame (modify this part according to your needs)
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB
    #frame = model.render()[0]  #

    # Draw bounding boxes on the frame (modify this part according to your needs)
    for det in detections:
        ##box = det[:4].int().cpu().numpy()
        #box = det[:4].cpu().numpy().astype(int)

        #rounded_box = round_box(box)
        #rounded_box = list(map(int, box))  
        #rounded_box = np.round(det[:4])
        #rounded_box = np.array(rounded_box, dtype=int)
        #rounded_box = np.round(det[:4]).astype(int)

        #rounded_box = np.rint(det[:4]).astype(int)

        rounded_box = np.round(det[:4]).astype(int)


        cv2.rectangle(frame, (rounded_box[0], rounded_box[1]), (rounded_box[2], rounded_box[3]), (0, 255, 0), 2)        

        #cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)


    ### Draw bounding boxes on the frame (modify this part according to your needs)
    #frame = results.render()[0]

    ### Write the frame to the output video
    output_video.write(frame)

    ### Increment frame count
    frame_count += 1

    # Display FPS every 100 frames
    if frame_count % 100 == 0:
        fps = frame_count / (time.time() - fps_start_time)
        print(f'FPS: {fps:.2f}')

# Release video capture and writer
cap.release()
output_video.release()

# Calculate and print average FPS
average_fps = frame_count / (time.time() - fps_start_time)
print(f'Average FPS: {average_fps:.2f}')
