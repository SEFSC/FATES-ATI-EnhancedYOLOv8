from ultralytics import YOLO
import torch
import os
import glob
import numpy as np
from sklearn.metrics import average_precision_score
from collections import defaultdict

# Check if CUDA is available and set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the model
model = YOLO('/work/cshah/updatedYOLOv8/ultralytics/runs/detect/train176/weights/best.pt')
model.model.to(device)


# Path to the test images and ground truth annotations
test_images_path = '/work/cshah/datasets/GFISHERD24/images/test2007'
ground_truth_path = '/work/cshah/datasets/GFISHERD24/labels/test2007'  

# Prepare ground truth data (example structure)
ground_truths = defaultdict(list)  # For storing ground truth bounding boxes and labels
for annotation_file in glob.glob(os.path.join(ground_truth_path, '*.txt')):
    with open(annotation_file, 'r') as f:
        # Parse annotations (assume format: class_id x_center y_center width height)
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            bbox = list(map(float, parts[1:]))
            ground_truths[annotation_file].append((class_id, bbox))

# Perform inference on the test set
test_images = glob.glob(os.path.join(test_images_path, '*.jpg'))
predictions = defaultdict(list)  # For storing predictions

for image_path in test_images:
    results = model(image_path)
    # Inspect results
    print(dir(results))  # Print attributes of results object
    for result in results:
        boxes = result.boxes  # Check the attribute used for bounding boxes
        for box in boxes:
            # Example attributes; adjust as necessary based on inspection
            class_id = int(box.cls)
            bbox = box.xyxy  # Get the bounding box coordinates
            predictions[image_path].append((class_id, bbox))

# Compute metrics (e.g., Average Precision)
def compute_ap(preds, gts, num_classes):
    aps = []
    class_aps = {}
    for cls in range(num_classes):
        y_true = []
        y_scores = []
        for img_path in preds:
            pred_bboxes = [bbox for (cid, bbox) in preds[img_path] if cid == cls]
            gt_bboxes = [bbox for (cid, bbox) in gts[img_path] if cid == cls]
            # For simplicity, assume 1-to-1 matching; adjust as needed
            for bbox in gt_bboxes:
                y_true.append(1)
                y_scores.append(1.0 if bbox in pred_bboxes else 0.0)
            for bbox in pred_bboxes:
                if bbox not in gt_bboxes:
                    y_true.append(0)
                    y_scores.append(1.0)
        if y_true:
            ap = average_precision_score(y_true, y_scores)
            class_aps[cls] = ap
            aps.append(ap)
        else:
            class_aps[cls] = 0.0  # No ground truth or predictions for this class

    mAP = np.mean(aps) if aps else 0
    return mAP, class_aps

# Example with dummy number of classes
num_classes = 80  # Set this according to your dataset
mAP, class_aps = compute_ap(predictions, ground_truths, num_classes)

print(f"Mean Average Precision (mAP): {mAP:.4f}")
print("Average Precision per Class:")
for cls, ap in class_aps.items():
    print(f"Class {cls}: {ap:.4f}")
print(f"Mean Average Precision (mAP): {mAP:.4f}")
print("Average Precision per Class:")
for cls, ap in class_aps.items():
    print(f"Class {cls}: {ap:.4f}")
