from ultralytics import YOLO
import yaml
import torch
torch.cuda.empty_cache()

import os

# Set environment variable to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Load a model
#model = YOLO("yolov8n.yaml")  # build a new model from scratch
#model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

#model = YOLO("yolov8s.yaml")  # build a new model from scratch
#model = YOLO("yolov8s.pt")

from ultralytics import YOLO
import yaml
import torch

def load_and_modify_model(yaml_path, weights_path):
    # Load the modified configuration from yolov8m.yaml
    with open(yaml_path, 'r') as file:
        modified_config = yaml.safe_load(file)

    # Load the pre-trained model
    #model = YOLO("yolov8m.pt")
    model = YOLO("/work/cshah/updatedYOLOv8/ultralytics/runs/detect/train74/weights/best.pt")



    # Move model to CPU
    model = model.to('cpu')

    # Replace layers or modify the model as needed
    for layer_name, new_layer_config in modified_config.items():
        if layer_name in model.model.state_dict():
            model.model.state_dict()[layer_name] = new_layer_config

    # Save the modified model configuration (if needed)
    with open("/work/cshah/updatedYOLOv8/ultralytics/ultralytics/cfg/models/v8/modified_yolov8m.yaml", 'w') as file:
        yaml.safe_dump(modified_config, file)

    # Move model back to GPU
    model = model.to('cuda')

    return model

yaml_path = '/work/cshah/updatedYOLOv8/ultralytics/ultralytics/cfg/models/v8/yolov8.yaml'
#weights_path = 'path/to/yolov8m.pt'
weights_path = '/work/cshah/updatedYOLOv8/ultralytics/runs/detect/train74/weights/best.pt'

# Load and modify the model
model = load_and_modify_model(yaml_path, weights_path)

# Print the new model layers
print(model.model)



#model = YOLO("/work/cshah/updatedYOLOv8/ultralytics/runs/detect/train74/weights/best.pt")


## Load the modified configuration from yolov8m.yaml
#with open("/work/cshah/updatedYOLOv8/ultralytics/ultralytics/cfg/models/v8/yolov8.yaml", 'r') as file:
#    modified_config = yaml.safe_load(file)

## Replace layers or modify the model as needed
## This is a simplified example. Adjust based on the actual structure and requirements.
#for layer_name, new_layer_config in modified_config.items():
#    if layer_name in model.model.state_dict():
#        model.model.state_dict()[layer_name] = new_layer_config

## Save the modified model configuration (if needed)
#with open("/work/cshah/updatedYOLOv8/ultralytics/ultralytics/cfg/models/v8/modified_yolov8m.yaml", 'w') as file:
#    yaml.safe_dump(modified_config, file)

## Rebuild the model using the modified configuration
#model = YOLO("/work/cshah/updatedYOLOv8/ultralytics/ultralytics/cfg/models/v8/modified_yolov8m.yaml")

#model = YOLO("yolov8m.yaml")
#model = YOLO("yolov8x.yaml")  # build a new model from scratch
#model = YOLO(' ')
#model = YOLO("yolov8m.pt")

##pretrained weight for C3TR in YOLOv8
#model = YOLO("/work/cshah/YOLOv8/runs/detect/train34/weights/best.pt") ##pret
#model = YOLO("/work/cshah/updatedYOLOv8/ultralytics/runs/detect/train15/weights/best.pt")

#model = YOLO("/work/cshah/updatedYOLOv8/ultralytics/runs/detect/train74/weights/best.pt")

##weight trained on seamapd21
#model = YOLO("/work/cshah/updatedYOLOv8/ultralytics/runs/detect/train126/weights/best.pt")

##after more epochs
#model = YOLO("/work/cshah/updatedYOLOv8/ultralytics/runs/detect/train19/weights/best.pt")


#model = YOLO("yolov8l.yaml")  # build a new model from scratch
#model = YOLO("yolov8l.pt")  

## Use the model
#model.train(data="coco128.yaml", epochs=50)  # train the model
#model.train(data="pasca_data_old.yaml", epochs=300 )

from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# Example of using mixed precision in a training loop
for data, target in dataloader:
    with autocast():
        output = model(data)
        loss = loss_function(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# When training or evaluating, reduce the batch size
batch_size = 8  # Reduce this number if memory issues persist
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Enable gradient checkpointing
model.model.gradient_checkpointing_enable()


import os
import torch
from ultralytics import YOLO
import yaml
from torch.cuda.amp import autocast, GradScaler

# Clear GPU memory
torch.cuda.empty_cache()

# Set environment variable to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def load_and_modify_model(yaml_path, weights_path):
    # Load the modified configuration from yolov8m.yaml
    with open(yaml_path, 'r') as file:
        modified_config = yaml.safe_load(file)

    # Load the pre-trained model
    #model = YOLO("yolov8m.pt")
    model = YOLO("/work/cshah/updatedYOLOv8/ultralytics/runs/detect/train74/weights/best.pt")



    # Move model to CPU
    model = model.to('cpu')

    # Replace layers or modify the model as needed
    for layer_name, new_layer_config in modified_config.items():
        if layer_name in model.model.state_dict():
            model.model.state_dict()[layer_name] = new_layer_config

    # Save the modified model configuration (if needed)
    with open("/work/cshah/updatedYOLOv8/ultralytics/ultralytics/cfg/models/v8/modified_yolov8m.yaml", 'w') as file:
        yaml.safe_dump(modified_config, file)

    # Move model back to GPU
    model = model.to('cuda')

    return model

yaml_path = '/work/cshah/updatedYOLOv8/ultralytics/ultralytics/cfg/models/v8/yolov8.yaml'
#weights_path = 'path/to/yolov8m.pt'
weights_path = '/work/cshah/updatedYOLOv8/ultralytics/runs/detect/train74/weights/best.pt'



# Load and modify the model
model = load_and_modify_model(yaml_path, weights_path)

# Print the new model layers
print(model.model)

# Example of using mixed precision in a training loop
scaler = GradScaler()
batch_size = 8  # Adjust as needed
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for data, target in dataloader:
    with autocast():
        output = model(data)
        loss = loss_function(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()


#model.train(data="pasca_data_nabi.yaml", epochs=300 )
#model.train(data="pasca_data_GFISHERD24.yaml", epochs=300 )
#model.train(data="pasca_data_nabi.yaml", epochs=300, resume =True )

#model.train(data="pasca_data.yaml", epochs=300, mode = "train resume")  # train the model
#metrics = model.val() 
