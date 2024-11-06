from ultralytics import YOLO
import os
import cv2
import torch
import csv

# Check if CUDA is available and set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


## Directories
#video_dir = '/work/cshah/unseen_videos_jackonedrive/2019 Sojo Video'
#frame_dir = '/work/cshah/extracted_frames_yolo_19Sojo'
#output_csv_path = '/work/cshah/frame_counts_unseen_videos/2019_Sojo_Video.csv'

import csv

# Path to the CSV file
csv_file_path = '/work/cshah/generated_csv_yolo/2019_Sojo_Video/941902239_cam1_detections.csv'

# Dictionary to store the count of unique frame identifiers
identifier_counts = {}

# Read the CSV file
with open(csv_file_path, 'r') as csvfile:
    csv_reader = csv.reader(csvfile)
    header = next(csv_reader)  # Skip the header row

    for row in csv_reader:
        identifier = row[0]  # Assuming the unique identifier is in the first column
        if identifier not in identifier_counts:
            identifier_counts[identifier] = 1
        else:
            identifier_counts[identifier] += 1

# Calculate the total number of unique frames
total_unique_frames = len(identifier_counts)
total_frames = sum(identifier_counts.values())

# Print the results
print(f"Total number of unique frame identifiers: {total_unique_frames}")
print(f"Total number of frames: {total_frames}")

# Optionally, save the results to a new CSV file
output_csv_path = '/work/cshah/frame_counts_unseen_videos/sampleframecount.csv'
##output_csv_path = '/path/to/output/csvfile.csv'
csv_header = ['Unique Identifier', 'Frame Count']

with open(output_csv_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(csv_header)
    for identifier, count in identifier_counts.items():
        csv_writer.writerow([identifier, count])

print("Frame counts saved to:", output_csv_path)
