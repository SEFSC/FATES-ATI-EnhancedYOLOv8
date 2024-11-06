from ultralytics import YOLO

from ultralytics import YOLO
import torch

# Check if CUDA is available and set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

##Optimal model weight on SEAMAPD23
# Load the model
model = YOLO('/work/cshah/updatedYOLOv8/ultralytics/runs/detect/train176/weights/best.pt')
model.model.to(device)
#metrics = model.test()

#model.train(data="pasca_data_nabi.yaml", epochs=300 )
#model.train(data="pasca_data_GFISHERD24.yaml", epochs=300 )
#model.train(data="pasca_data_nabi.yaml", epochs=300, resume =True )

#model.train(data="pasca_data.yaml", epochs=300, mode = "train resume")  # train the model
#metrics = model.val() 
metrics = model.val(data='pasca_data_GFISHERD24.yaml', device=0,split='test',save_json=True)
##metrics = model.val(data='pasca_data_nabi.yaml', device=0,split='test',save_json=True)

