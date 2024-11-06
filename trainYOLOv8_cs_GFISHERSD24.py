from ultralytics import YOLO
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize model with correct configuration
model = YOLO("yolov8x.yaml")

# Load pretrained weights
model.load_state_dict(torch.load("/work/cshah/updatedYOLOv8/ultralytics/runs/detect/train74/weights/best.pt"))

# Ensure model is in training mode
model.train()

# Move model to the appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# If using DDP, wrap the model
if torch.cuda.device_count() > 1:
    model = DDP(model, device_ids=[0, 1, 2, 3], find_unused_parameters=False)

# Train the model
model.train(data="pasca_data_GFISHERD24.yaml", epochs=300)

# Validate the model
metrics = model.val()
print("Validation Metrics:", metrics)

# Test the model
###test_metrics = model.test()
test_metrics = model.val(data='pasca_data_GFISHERD24.yaml', device=0,split='test',save_json=True)
print("Test Metrics:", test_metrics)
