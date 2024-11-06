import cv2

def change_fps(input_video_path, output_video_path, target_fps):
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    
    # Get original properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, target_fps, (frame_width, frame_height))
    
    # Read and write frames
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frame_count += 1
    
    cap.release()
    out.release()

# Adjust FPS of input video to 5
change_fps('/work/cshah/mmtracking_cshah/data/GFISHERS24/SC2-camera3_03-13-21_13-41-24.000.avi', '/work/cshah/updatedYOLOv8/ultralytics/adjusted_SC2-camera3_03-13-21_13-41-24.000.avi', target_fps=5)
