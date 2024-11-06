from ultralytics import YOLO

imgsz=1280 

# Load a model
model = YOLO("yolov8x.yaml")

##pretrained weight for C3TR in YOLOv8
#model = YOLO("/work/cshah/updatedYOLOv8/ultralytics/runs/detect/train74/weights/best.pt")
model = YOLO("/scratch/morrill/users/cshah/updatedYOLOv8/ultralytics/runs/detect/train74/weights/best.pt")



#model.train(data="pasca_data_GFISHERD24_SHAH.yaml", epochs=300,imgsz=1280)
model.train(data="pasca_data_GFISHERD24_SHAH.yaml", epochs=200,imgsz=1280)
#model.train(data="pasca_data_GFISHERD24_SHAH_New.yaml", epochs=300)
#model.train(data="pasca_data_GFISHERD24_SHAH.yaml", epochs=200)    # train the model

metrics = model.val() 

test_metrics = model.val(data='pasca_data_GFISHERD24_SHAH.yaml', split='test',save_json=True)
#test_metrics = model.val(data='pasca_data_GFISHERD24_SHAH_New.yaml', split='test',save_json=True)
