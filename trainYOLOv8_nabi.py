from ultralytics import YOLO

# Load a model
#model = YOLO("yolov8n.yaml")  # build a new model from scratch
#model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

#model = YOLO("yolov8s.yaml")  # build a new model from scratch
#model = YOLO("yolov8s.pt")

#model = YOLO("yolov8m.yaml")
model = YOLO("yolov8x.yaml")

#model = YOLO("yolov8_trans.yaml")

#model = YOLO("yolov8x.yaml")  # build a new model from scratch
#model = YOLO(' ')
#model = YOLO("yolov8m.pt")
##pretrained weight for C3TR in YOLOv8
#model = YOLO("/work/cshah/YOLOv8/runs/detect/train34/weights/best.pt") ##pret
# model = YOLO("/work/cshah/updatedYOLOv8/ultralytics/runs/detect/train15/weights/best.pt")
model = YOLO("/work/cshah/updatedYOLOv8/ultralytics/runs/detect/train74/weights/best.pt")


##after more epochs
#model = YOLO("/work/cshah/updatedYOLOv8/ultralytics/runs/detect/train19/weights/best.pt")


#model = YOLO("yolov8l.yaml")  # build a new model from scratch
#model = YOLO("yolov8l.pt")  

# Use the model
#model.train(data="coco128.yaml", epochs=50)  # train the model
#model.train(data="pasca_data_old.yaml", epochs=300 )

#model.train(data="pasca_data_nabi.yaml", epochs=300 )
#model.train(data="pasca_data_GFISHERD24.yaml", epochs=300 )


#from torch.nn.parallel import DistributedDataParallel as DDP

# Assuming model is your model instance
#ddp_model = DDP(model, find_unused_parameters=False)

# Assuming model is your model insta
##Distributed training on multiple nodes
model.train(data="pasca_data_GFISHERD24.yaml", epochs=200)
#ddp_model.train(data="pasca_data_GFISHERD24.yaml", epochs=300, device=[1,2,3,4] )



#model.train(data="pasca_data_nabi.yaml", epochs=300, resume =True )

#model.train(data="pasca_data.yaml", epochs=300, mode = "train resume")  # train the model
metrics = model.val() 


test_metrics = model.val(data='pasca_data_GFISHERD24.yaml', split='test',save_json=True)
print("Test Metrics:", test_metrics)
