from ultralytics import YOLO
import torch

# Check if CUDA is available and set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the model
model = YOLO('/work/cshah/updatedYOLOv8/ultralytics/runs/detect/train176/weights/best.pt')
model.model.to(device)

# Perform tracking with the model
#results = model.track(source="/work/cshah/updatedYOLOv8/ultralytics/adjusted_SC2-camera3_03-13-21_13-41-24.000.avi", save=True, conf=0.5, iou=0.5, show=True, tracker="bytetrack.yaml")
results = model.track(source="/work/cshah/mmtracking_cshah/data/GFISHERS24/SC2-camera3_03-13-21_13-41-24.000.avi", save=True, conf=0.5, iou=0.5, show=True, tracker="bytetrack.yaml")


# Read ground truth file and extract frame numbers
def extract_frames_from_gt(gt_file_path):
    frame_numbers = set()
    with open(gt_file_path, 'r') as f:
        for line in f:
            parts = line.split()
            if parts:
                frame_numbers.add(int(parts[0]))  # Assuming frame number is the first item
    return frame_numbers

# Define path to ground truth file
gt_file_path = 'groundtruthMOT.txt'
frames_with_gt = extract_frames_from_gt(gt_file_path)

# Save only the relevant frames' predictions
with open('filtered_track_results_yolo_mot.txt', 'w') as f:
    for frame_id, result in enumerate(results):
        if frame_id + 1 in frames_with_gt:
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
