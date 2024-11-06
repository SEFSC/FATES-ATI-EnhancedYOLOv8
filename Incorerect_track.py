import motmetrics as mm
import pandas as pd
import numpy as np

# Define the paths to the ground truth and predicted tracking files
gt_txt_path = '/work/cshah/updatedYOLOv8/ultralytics/groundtruthMOT.txt'
pred_txt_path = '/work/cshah/updatedYOLOv8/ultralytics/track_results_yolo_mot.txt'

# Load ground truth and YOLO predictions
gt = mm.io.loadtxt(gt_txt_path, fmt='mot15-2D')
pred = mm.io.loadtxt(pred_txt_path, fmt='mot15-2D')

# Inspect column names and first few rows
print("Ground Truth Columns:", gt.columns)
print("Ground Truth Sample Data:\n", gt.head())
print("Predictions Columns:", pred.columns)
print("Predictions Sample Data:\n", pred.head())

# Extract relevant levels if index is a MultiIndex
if isinstance(gt.index, pd.MultiIndex):
    gt_frame_index = gt.index.get_level_values(0).astype(int)
else:
    gt_frame_index = gt.index.astype(int)

if isinstance(pred.index, pd.MultiIndex):
    pred_frame_index = pred.index.get_level_values(0).astype(int)
else:
    pred_frame_index = pred.index.astype(int)

# Extract class_id from the last column in ground truth and predictions
gt['class_id'] = gt.iloc[:, -1].replace({np.nan: -1, np.inf: -1, -np.inf: -1}).astype(int)
pred['class_id'] = pred.iloc[:, -1].replace({np.nan: -1, np.inf: -1, -np.inf: -1}).astype(int)

# Ensure consistent class IDs
# Map class IDs from predictions to those used in ground truth
class_id_mapping = {
    82: 1,  # Example mapping, adjust according to your ground truth
    123: 2,
    140: 3
}

pred['class_id'] = pred['class_id'].map(class_id_mapping).fillna(-1).astype(int)

# Create a single accumulator for all classes
accumulator = mm.MOTAccumulator(auto_id=True)

# Get the range of frames
all_frames = sorted(set(gt_frame_index).union(pred_frame_index))
print(f"Number of frames: {len(all_frames)}")

# Process each frame for all objects as a single class
for frame in all_frames:
    # Get ground truth and predicted detections for the current frame
    gt_dets = gt.loc[frame].values[:, :4] if frame in gt_frame_index else []
    pred_dets = pred.loc[frame].values[:, :4] if frame in pred_frame_index else []
    
    # Ensure gt_dets and pred_dets are NumPy arrays
    gt_dets = np.array(gt_dets) if len(gt_dets) > 0 else np.zeros((0, 4))
    pred_dets = np.array(pred_dets) if len(pred_dets) > 0 else np.zeros((0, 4))
    
    # Debugging prints
    print(f"Frame {frame}: Ground Truth Detections:\n{gt_dets}")
    print(f"Frame {frame}: Predicted Detections:\n{pred_dets}")
    
    # Compute IoU matrix
    if gt_dets.size > 0 and pred_dets.size > 0:
        dists = mm.distances.iou_matrix(gt_dets, pred_dets, max_iou=0.5)
    else:
        dists = np.zeros((len(gt_dets), len(pred_dets)))
    
    # Print IoU matrix for debugging
    print(f"Frame {frame}: IoU Matrix:\n{dists}")
    
    # Update the accumulator
    accumulator.update(
        list(range(len(gt_dets))),
        list(range(len(pred_dets))),
        dists
    )

# Compute metrics for the single class
mh = mm.metrics.create()
summary = mh.compute(accumulator, metrics=mm.metrics.motchallenge_metrics, name='single_class')
print("Metrics for single class:\n", summary)

# Print available metrics
print(mh.list_metrics_markdown())
