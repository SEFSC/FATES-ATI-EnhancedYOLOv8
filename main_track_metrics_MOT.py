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
#results = model.track(source="/work/cshah/unseen_videos_jackonedrive/2022 Pisces Video/762201004_cam3.avi", save=True, conf=0.5, iou=0.5, show=True, tracker="bytetrack.yaml")
#results = model.track(source="/work/cshah/mmtracking_cshah/data/GFISHERS24/SC2-camera3_03-13-21_13-41-24.000.avi", save=True, conf=0.5, iou=0.5, show=True, tracker="bytetrack.yaml")
results = model.track(source="/work/cshah/updatedYOLOv8/ultralytics/adjusted_SC2-camera3_03-13-21_13-41-24.000.avi", save=True, conf=0.5, iou=0.5, show=True, tracker="bytetrack.yaml")

with open('track_results_yolo_mot.txt', 'w') as f:
    for frame_id, result in enumerate(results):
        for box in result.boxes:
            bbox = box.xyxy[0].tolist()  # Convert from tensor to list
            track_id = box.id  # Get track id

            if track_id is not None:
                track_id = track_id.item()  # Ensure track_id is valid before accessing .item()

                conf = box.conf.item()  # Get confidence score
                f.write(f'{frame_id+1},{track_id},{bbox[0]},{bbox[1]},{bbox[2]-bbox[0]},{bbox[3]-bbox[1]},-1,-1,{conf}\n')
            else:
                # Handle the case where track_id is None (optional: log, skip, etc.)
                print(f"Warning: No track ID for frame {frame_id+1}")


## Save results in MOTChallenge format (frame, id, bbox, conf)
#with open('track_results_yolo_mot.txt', 'w') as f:
#    for frame_id, result in enumerate(results):
#        for box in result.boxes:
#            bbox = box.xyxy[0].tolist()  # Convert from tensor to list
#            track_id = box.id.item()  # Get track id
#            conf = box.conf.item()  # Get confidence score
#            f.write(f'{frame_id+1},{track_id},{bbox[0]},{bbox[1]},{bbox[2]-bbox[0]},{bbox[3]-bbox[1]},-1,-1,{conf}\n')


### Save the tracking results to a file
#with open('tracking_results.txt', 'w') as f:
#    for frame_results in results:
#        for track_id, bbox in enumerate(frame_results.boxes.data):
#            if len(bbox) == 6:
#                x1, y1, x2, y2, conf, _ = bbox
#            elif len(bbox) == 7:
#                x1, y1, x2, y2, conf, _, _ = bbox
#            else:
##                # Handle unexpected case
#                pass
##            #x1, y1, x2, y2, conf,_, _ = bbox
#            f.write(f"{track_id + 1} {x1} {y1} {x2 - x1} {y2 - y1} {conf}\n")



