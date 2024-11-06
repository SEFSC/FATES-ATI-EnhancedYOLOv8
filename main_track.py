from ultralytics import YOLO
import torch

## Check if CUDA is available and set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

## Load an official or custom model
#model = YOLO('best.pt')  # Load an official Detect model
model = YOLO('/work/cshah/updatedYOLOv8/ultralytics/runs/detect/train176/weights/best.pt')
model.model.to(device)


# Perform tracking with the model
#results = model.track(source="fish.avi", save=True, conf=0.5, iou=0.5, show=True, tracker="bytetrack.yaml")  # Tracking with default tracker
#results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml")  # Tracking with ByteTrack tracker

results = model.track(source="/work/cshah/unseen_videos_jackonedrive/2022 Pisces Video/762201029_cam3.avi", save=True, conf=0.5, iou=0.5, show=True, tracker="bytetrack.yaml")  # Tracking with default tracker
#print('track results',results)
