from ultralytics import YOLO
import torch

# Check if CUDA is available and set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

## Load an official or custom model
#model = YOLO('best.pt')  # Load an official Detect model
model = YOLO('/work/cshah/updatedYOLOv8/ultralytics/runs/detect/train176/weights/best.pt')
model.model.to(device)


## Perform tracking with the model
#results = model.track(source="2021_NCD-038c.mp4", save=True, conf=0.5, iou=0.5, show=True, tracker="bytetrack.yaml")
results = model.track(source="/work/cshah/unseen_videos_jackonedrive/2022 Pisces Video/762201004_cam3.avi", save=True, conf=0.5, iou=0.5, show=True, tracker="bytetrack.yaml")


# Save the tracking results to a file
with open('tracking_results.txt', 'w') as f:
    for frame_results in results:
        for track_id, bbox in enumerate(frame_results.boxes.data):
            if len(bbox) == 6:
                x1, y1, x2, y2, conf, _ = bbox
            elif len(bbox) == 7:
                x1, y1, x2, y2, conf, _, _ = bbox
            else:
                # Handle unexpected case
                pass

            #x1, y1, x2, y2, conf,_, _ = bbox
            f.write(f"{track_id + 1} {x1} {y1} {x2 - x1} {y2 - y1} {conf}\n")

