import cv2
import os
import csv

# Directories
#video_dir = '/work/cshah/unseen_videos_jackonedrive/2019 Sojo Video'
#csv_dir = '/work/cshah/generated_csv_yolo/2019_Sojo_Video'
#output_csv_path = '/work/cshah/frame_counts_unseen_videos/2019_Sojo_Video_final.csv'
#video_dir = '/work/cshah/unseen_videos_jackonedrive/2019 Pisces Video'
#csv_dir = '/work/cshah/generated_csv_yolo/2019_Pisces_Video'
#output_csv_path = '/work/cshah/frame_counts_unseen_videos/2019_Pisces_Video_final.csv'
#video_dir = '/work/cshah/unseen_videos_jackonedrive/2021 Pisces'
#csv_dir = '/work/cshah/generated_csv_yolo/2021_Pisces'
#output_csv_path = '/work/cshah/frame_counts_unseen_videos/2021_Pisces_Video_final.csv'
#video_dir = '/work/cshah/unseen_videos_jackonedrive/2022 Pisces Video'
#csv_dir = '/work/cshah/generated_csv_yolo/2022_Pisces_video'
#output_csv_path = '/work/cshah/frame_counts_unseen_videos/2022_Pisces_Video_final.csv'
#video_dir = '/work/cshah/unseen_videos_jackonedrive/2022 Sojo Video_done'
video_dir = '/work/cshah/unseen_videos_jackonedrive/2022 Sojo Video_done/try'
csv_dir = '/work/cshah/generated_csv_yolo/2022_Sojo_Video'
output_csv_path = '/work/cshah/frame_counts_unseen_videos/2022_Sojo_942205003_cam3.csv'
#output_csv_path = '/work/cshah/frame_counts_unseen_videos/2022_Sojo_Video_final.csv'


# Frame extraction settings
fps_extract = 5  # Frames per second to extract

# Dictionary to store the count of unique frame identifiers and total frames from videos
unique_frame_identifiers = set()
identifier_counts = {}
video_frame_counts = {}

# Process CSV files to get unique frame identifiers
for csv_file in os.listdir(csv_dir):
    if csv_file.endswith('.csv'):
        csv_file_path = os.path.join(csv_dir, csv_file)
        
        try:
            # Read the CSV file
            with open(csv_file_path, 'r') as csvfile:
                csv_reader = csv.reader(csvfile)
                header = next(csv_reader)  # Skip the header row

                for row in csv_reader:
                    identifier = row[2]  # Assuming the unique identifier is in the first column
                    unique_identifier = f"{csv_file}_{identifier}"  # Combine video file name and identifier
                    unique_frame_identifiers.add(unique_identifier)
                    if identifier not in identifier_counts:
                        identifier_counts[identifier] = 1
                    else:
                        identifier_counts[identifier] += 1
        except Exception as e:
            print(f"Error reading file {csv_file}: {e}")

# Calculate total number of unique frames and total frames from CSV
total_unique_frames = len(unique_frame_identifiers)
total_frames_from_csv = sum(identifier_counts.values())

# Process videos to count the total number of frames at 5 FPS
for video_file in os.listdir(video_dir):
    if video_file.endswith('.avi'):  # Adjust extension if needed
        # Extract the base name of the video file without extension
        base_name = os.path.splitext(video_file)[0]
        
        # Corresponding CSV file
        csv_file_name = f"{base_name}_detections.csv"
        csv_file_path = os.path.join(csv_dir, csv_file_name)

        # Check if the corresponding CSV file exists
        if not os.path.isfile(csv_file_path):
            print(f"Skipping video file {video_file} as corresponding CSV file {csv_file_name} is not available.")
            continue

        video_path = os.path.join(video_dir, video_file)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Could not open video file {video_file}.")
            continue

        # Get the frame rate of the video
        fps_video = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(round(fps_video / fps_extract))  # Calculate interval based on target FPS

        frame_count = 0
        extracted_frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Extract frame at specified intervals
            if frame_count % frame_interval == 0:
                extracted_frame_count += 1

            frame_count += 1

        video_frame_counts[video_file] = extracted_frame_count

        cap.release()

# Calculate total number of frames from videos at 5 FPS
total_frames_from_videos = sum(video_frame_counts.values())

# Print results
print(f"Total number of unique frame identifiers (from CSV): {total_unique_frames}")
print(f"Total number of frames (from CSV): {total_frames_from_csv}")
print(f"Total number of frames (from videos at {fps_extract} FPS): {total_frames_from_videos}")

# Save the results to a new CSV file
csv_header = ['Metric', 'Count']
results = [
    ['Total number of unique frame identifiers (from CSV)', total_unique_frames],
    ['Total number of frames (from CSV)', total_frames_from_csv],
    ['Total number of frames (from videos at 5 FPS)', total_frames_from_videos]
]

with open(output_csv_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(csv_header)
    csv_writer.writerows(results)

print("Results saved to:", output_csv_path)
