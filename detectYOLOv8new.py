import torch
from pathlib import Path
from PIL import Image
from IPython.display import Image as IPImage, display
from ultralytics import YOLO

# Set the path to your YOLOv8 model weights file (e.g., 'yolov8.weights')
weights_path = '/work/cshah/YOLOv8/runs/detect/train265/weights/best.pt'

## Set the path to your input images directory
input_images_path = '../../datasets/images/test/'
##input_images_path = '../../datasets/images/test/YSC4_Camera4_08-07-19_19-01-400004.png'



# Set the path to the output directory where you want to save the annotated images
output_images_path = '/work/cshah/updatedYOLOv8/ultralytics/detectedresults'
Path(output_images_path).mkdir(parents=True, exist_ok=True)

# Load YOLOv8 model
#model = torch.hub.load('ultralytics/yolov5:v6.0', 'yolov5s', pretrained=False)
#model.load_state_dict(torch.load(weights_path))

model = YOLO('/work/cshah/YOLOv8/runs/detect/train265/weights/best.pt')

#model.eval()

# Get a list of input images
image_files = list(Path(input_images_path).rglob('*.png'))

# Perform inference and save annotated images
for image_file in image_files:
    # Load input image
    img = Image.open(image_file)

    # Perform inference
    results = model(img)

    # Save annotated image
    annotated_image_path = Path(output_images_path) / image_file.name
    results.save(annotated_image_path)
    ##model.save_img(annotated_image_path, results)

print(f"Annotated images saved in the folder: {output_images_path}")
