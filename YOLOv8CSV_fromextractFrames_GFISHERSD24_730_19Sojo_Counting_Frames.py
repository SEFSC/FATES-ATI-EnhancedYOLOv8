import csv
import os

# Directory containing the CSV files
#csv_dir = '/work/cshah/generated_csv_yolo/2019_Sojo_Video'
#csv_dir = '/work/cshah/generated_csv_yolo/2019_Pisces_Video'
#csv_dir = '/work/cshah/generated_csv_yolo/2021_Pisces'
#csv_dir = '/work/cshah/generated_csv_yolo/2022_Pisces_video'
csv_dir = '/work/cshah/generated_csv_yolo/2022_Sojo_Video'

# Set to store unique frame identifiers
unique_frame_identifiers = set()

# Iterate over all CSV files in the directory
csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
print(f"Processing {len(csv_files)} CSV files...")

for filename in csv_files:
    csv_file_path = os.path.join(csv_dir, filename)
    print(f"Reading file: {filename}")
    
    try:
        # Read the CSV file
        with open(csv_file_path, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            header = next(csv_reader)  # Skip the header row

            for row in csv_reader:
                if row:  # Check if the row is not empty
                    identifier = row[0].strip()  # Assuming the unique identifier is in the first column
                    # Combine video filename with identifier to create a unique key
                    unique_key = f"{filename}_{identifier}"
                    unique_frame_identifiers.add(unique_key)
    except Exception as e:
        print(f"Error reading file {filename}: {e}")

# Calculate the total number of unique frames
total_unique_frames = len(unique_frame_identifiers)

# Print the results
print(f"Total number of unique frame identifiers (considering multiple detections as one): {total_unique_frames}")

# Save the results to a new CSV file
#output_csv_path = '/work/cshah/frame_counts_unseen_videos/2019_Sojo_Video_combined.csv'
#output_csv_path = '/work/cshah/frame_counts_unseen_videos/2019_Pisces_Video_combined.csv'
#output_csv_path = '/work/cshah/frame_counts_unseen_videos/2021_Pisces_Video_combined.csv'
#output_csv_path = '/work/cshah/frame_counts_unseen_videos/2022_Pisces_Video_combined.csv'
output_csv_path = '/work/cshah/frame_counts_unseen_videos/2022_Sojo_Video_combined.csv'

csv_header = ['Unique Identifier']

with open(output_csv_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(csv_header)
    for unique_key in unique_frame_identifiers:
        csv_writer.writerow([unique_key])

print("Unique frame identifiers saved to:", output_csv_path)
