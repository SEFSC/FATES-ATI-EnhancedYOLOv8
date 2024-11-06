import pandas as pd

# Load the predicted track into a DataFrame
pred = pd.read_csv('track_results_yolo_mot.txt', header=None)

# Assuming the frame_id is in the first column
pred_resampled = pred[pred[0] % 2 == 0]  # Keep every second frame

# Save the resampled prediction
pred_resampled.to_csv('track_results_yolo_mot_.txt', header=False, index=False)
