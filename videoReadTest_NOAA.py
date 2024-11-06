from google.cloud import storage
import os
import cv2
import numpy as np

URL = 'https://storage.googleapis.com/nmfs_odp_sefsc/PEMD/VIDEO_DATA/GOM_REEF_FISH/SoJo_2022/'
bucket = 'nmfs_odp_sefsc'
prefix = 'PEMD/VIDEO_DATA/GOM_REEF_FISH/SoJo_2022'

def list_files_from_bucket(bucket_name, prefix):
    """List the files in bucket_name containing the specified prefix. This assumes the bucket and contents
    are publically accessible without requiring login credentials

    Args:
        bucket_name: str, name of the Google Cloud storage bucket containing the desired files
        prefix: str, full directory chain in which the desired files reside. This is taken from the files'
            URL, starting after the bucket name and excluding the name of the file itself.

    Returns:
        List of file names satisfying the conditions above
    """
    # Public non-credentialed connection
    client = storage.Client.create_anonymous_client()
    # File names with 'prefix' contained in 'bucket_name'
    files = [blob.name for blob in client.list_blobs(bucket_name, prefix=prefix)]
    return files

files = list_files_from_bucket(bucket, prefix)
print(files)

cap = cv2.VideoCapture(os.path.join(URL, os.path.basename(files[0])))

print('starting to count frames')
#count frames
import cv2

# Target FPS for frame extraction
target_fps = 5

## Open the video file
#cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
else:
    # Get the original frame rate of the video
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    if fps_video == 0:
        print("Error: Could not retrieve FPS from video.")
    else:
        # Calculate the frame interval for extraction
        frame_interval = int(round(fps_video / target_fps))

        # Initialize counters
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

        print(f"Total number of frames at {target_fps} FPS: {extracted_frame_count}")

    # Release the video capture object
    cap.release()
