import cv2
import time
import torch
from ultralytics import YOLO
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


# Set up distributed training environment
dist.init_process_group(backend='nccl')
local_rank = dist.get_rank()
torch.cuda.set_device(local_rank)

#model = YOLO('/work/cshah/YOLOv8_weights_saved/YOLOv8s/weights/best.pt')

model = YOLO('/work/cshah/YOLOv8_weights_saved/YOLOv8l/weights/best.pt')

#model = YOLO('/work/cshah/YOLOv8_weights_saved/Yolov8m_enh_128batch/weights/best.pt')

#model = YOLO('/work/cshah/YOLOv8_weights_saved/YOLOvn/weights/best.pt')

# Wrap the model with DistributedDataParallel
model = DistributedDataParallel(model)

train_loader = get_distributed_train_loader()

### Load YOLOv5 model
#weights_path = 'path/to/your/yolov5/weights.pt'
#model = torch.hub.load('ultralytics/yolov5:v6.0', 'yolov5s', weights=weights_path).autoshape()

# Load video
#video_path = 'path/to/your/video.mp4'

#video_path = '../../datasets/2021TestVideo/762101178_cam3.avi'
#video_path = '../../datasets/2021TestVideo/762101001_cam3.avi'
#video_path = '../../datasets/2021TestVideo/762101002_cam3.avi'
#video_path = '../../datasets/2021TestVideo/762101525_cam3.avi'
#video_path = '../../datasets/2021TestVideo/762101457_cam3.avi'
#video_path = '../../datasets/2021TestVideo/762101449_cam3.avi'

video_path = '../../datasets/2021TestVideo/762101513_cam3.avi'  ##29 FPS
#video_path = '../../datasets/2021TestVideo/762101515_cam3.avi' ##31 FPS for frame count 1

cap = cv2.VideoCapture(video_path)

# Initialize variables for FPS calculation
frame_count = 0
total_detection_time = 0
    
###start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

   ## # Perform YOLOv5 detection on the frame
    start_time = time.time()
    results = model(frame)
    end_time = time.time()

    # Calculate detection time for the current frame
    detection_time = end_time - start_time

    # Accumulate total detection time
    total_detection_time += detection_time

    # Increment frame count
    frame_count += 1

    ##added to increase the detection time
    ##total_detection_time = total_detection_time*10e-3

    # Display FPS every 100 frames
    if frame_count % 1 == 0: ##100 originally
        average_fps = frame_count / total_detection_time
        print(f'Average FPS: {average_fps:.2f}')

# Release video capture
cap.release()

# Calculate and print final average FPS
average_fps = frame_count / total_detection_time
print(f'Final Average FPS: {average_fps:.2f}')
