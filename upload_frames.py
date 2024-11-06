import cv2
import os

def upload_frames(folder_path):
    # Get all file names in the folder
    files = os.listdir(folder_path)
    
    ### Sort files by name to ensure sequential order
    files.sort()
    
    # List to store frames
    frames = []
    
    for file in files:
        # Read each image file
        image_path = os.path.join(folder_path, file)
        frame = cv2.imread(image_path)
        
        # Append frame to the list
        if frame is not None:
            frames.append(frame)
            print("Uploaded:", file)  # Print the file nam
        else:
            print(f"Could not read image: {image_path}")
    
    return frames

# Example usage
folder_path = '/work/cshah/updatedYOLOv8/ultralytics/extracted_FRAMESn/'
frames = upload_frames(folder_path)

#print('First Frame name', frames[0])

## Show the first frame (assuming at least one frame was read)
#if frames:
#    cv2.imshow('First Frame', frames[0])
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#else:
#    print("No frames were read.")
