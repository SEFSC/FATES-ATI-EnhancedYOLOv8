import cv2
import os

#video_path = r'/data/mn918/data/videos_seamapd/762101028_cam3.avi'
#output_folder = r'/data/mn918/data/VOC2007/JPEGImages'

#video_path =r'../../datasets/2021TestVideo/762101515_cam3.avi'
video_path =r'../../datasets/2021TestVideo/762101178_cam3.avi'
output_folder = r'/work/cshah/updatedYOLOv8/ultralytics/extracted_FRAMESn'

frame_rate = 5  # Frames per second

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

frame_count = 0
while True:
    # Read a frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Check if it's time to save a frame based on the frame rate
    if frame_count % int(cap.get(cv2.CAP_PROP_FPS) / frame_rate) == 0:
        ### Save the frame as a PNG image with a filename like '762101028_cam3_1.png'
        #frame_filename = os.path.join(output_folder, f'762101028_cam3_{frame_count // int(cap.get(cv2.CAP_PROP_FPS) / frame_rate) + 1}.png')
        frame_filename = os.path.join(output_folder, f'762101178_cam3_{frame_count // int(cap.get(cv2.CAP_PROP_FPS) / frame_rate) + 1}.png')
        cv2.imwrite(frame_filename, frame)

    frame_count += 1

# Release the video capture object
cap.release()

print(f"Frames extracted and saved to {output_folder}")
