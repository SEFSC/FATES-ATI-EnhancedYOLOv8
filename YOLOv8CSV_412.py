from ultralytics import YOLO
import pprint
import os
import cv2
import csv
import torch
from pathlib import Path
# Importing the glob library
import glob 

#### Load a pretrained YOLOv8n model
##model = YOLO('yolov8n.pt')

#model = YOLO('/work/cshah/YOLOv8/runs/detect/train265/weights/best.pt')

#model = YOLO('/work/cshah/YOLOv8_weights_saved/YOLOv8l/weights/best.pt')

#model = YOLO('/work/cshah/YOLOv8_weights_saved/YOLOvn/weights/best.pt')

model = YOLO('/work/cshah/YOLOv8_weights_saved/Yolov8m_enh_128batch/weights/best.pt')


### Run inference on an image
#results = model('bus.jpg')

###results = model('../../datasets/images/test/', save=True)
#results = model('../../datasets/images/test/YSC4_Camera4_08-07-19_19-01-400004.png',save=True)
#results = model('../../datasets/2021TestVideo/762101449_cam3.avi',save=True)

testset = '/work/cshah/updatedYOLOv8/ultralytics/extracted_FRAMESn/'

##results = model('/work/cshah/updatedYOLOv8/ultralytics/extracted_FRAMES/762101028_cam3_69.png',save=True)
#results = model(testset, save=True)
### List files in the directory
list_test_files = os.listdir(testset)

###files_sort = list_test_files.sort()
# Get all files in the directory and sort the
#files_sort = sorted(os.listdir(directory))

files_sort = sorted(os.listdir(testset))

# Get all files in the directory using Pathlib
###files_sort = [file for file in Path(testset).iterdir() if file.is_file()]

##print('test files',list_test_files)
#print('sorted test files',files_sort)


#print('first image in sorted files',files_sort[0])
#print('last test files',list_test_files[7501])
#print('last test files',sortedfiles)

totalimags = len(list_test_files)
#print('total images in test sest',totalimags)

#print('name of testset in 1',testset[40])
#video_path = '../../datasets/2021TestVideo/762101449_cam3.avi'
#results = model('../../datasets/images/test//YSC4_Camera4_08-07-19_19-01-400004.png',save=True)


class_names= ['ACANTHURUSCOERULEUS', 'ACANTHURUS', 'ALECTISCILIARIS', 'ANISOTREMUSVIRGINICUS','ANOMURA','ANTHIINAE','ARCHOSARGUSPROBATOCEPHALUS','BALISTESCAPRISCUS',
       'BALISTESVETULA','BODIANUSPULCHELLUS','BODIANUSRUFUS','CALAMUSBAJONADO','CALAMUSLEUCOSTEUS','CALAMUSNODOSUS','CALAMUSPRORIDENS','CALAMUS','CANTHIDERMISSUFFLAMEN',
       'CANTHIGASTERROSTRATUS','CARANXBARTHOLOMAEI','CARANXCRYSOS','CARANXRUBER','CARCHARHINUSFALCIFORMIS','CARCHARHINUSPEREZI','CARCHARHINUSPLUMBEUS','CAULOLATILUSCHRYSOPS',
       'CAULOLATILUS CYANOPS','CAULOLATILUSCYANOPS','CENTROPRISTISOCYURA','CEPHALOPHOLISCRUENTATA','CEPHALOPHOLISFULVA','CHAETODONACULEATUS','CHAETODONCAPISTRATUS',
       'CHAETODONOCELLATUS','CHAETODONSEDENTARIUS','CHAETODON','CHROMISENCHRYSURUS','CHROMISINSOLATUS','CHROMIS','DERMATOLEPISINERMIS','DIODONTIDAE','DIPLECTRUMFORMOSUM',
       'DIPLECTRUM','EPINEPHELUSADSCENSIONIS','EPINEPHELUSFLAVOLIMBATUS','EPINEPHELUSMORIO','EPINEPHELUSNIGRITUS','EPINEPHELUS','EQUETUSLANCEOLATUS','EQUETUSUMBROSUS',
       'GONIOPLECTRUSHISPANUS','GYMNOTHORAXMORINGA','GYMNOTHORAXSAXICOLA','HAEMULONAUROLINEATUM','HAEMULONFLAVOLINEATUM','HAEMULONMACROSTOMUM','HAEMULONMELANURUM',
       'HAEMULONPLUMIERI','HALICHOERESBATHYPHILUS','HALICHOERESBIVITTATUS','HALICHOERESGARNOTI','HALICHOERES','HOLACANTHUSBERMUDENSIS','HOLACANTHUS','HOLANTHIUSMARTINICENSIS',
       'HOLOCENTRUS','HYPOPLECTRUSGEMMA','HYPOPLECTRUS','HYPOPLECTRUSUNICOLOR','IOGLOSSUS','KYPHOSUS','LACHNOLAIMUSMAXIMUS','LACTOPHRYSTRIGONUS','LIOPROPOMAEUKRINES',
       'LUTJANUSANALIS','LUTJANUSAPODUS','LUTJANUSBUCCANELA','LUTJANUSCAMPECHANUS','LUTJANUSGRISEUS','LUTJANUSSYNAGRIS','LUTJANUS','LUTJANUSVIVANUS','MALACANTHUSPLUMIERI',
       'MULLOIDICHTHYSMARTINICUS','MURAENARETIFERA','MYCTEROPERCABONACI','MYCTEROPERCAINTERSTIALIS','MYCTEROPERCAINTERSTITIALIS','MYCTEROPERCAMICROLEPIS','MYCTEROPERCAPHENAX',
       'MYCTEROPERCA','OCYURUSCHRYSURUS','OPHICHTHUSPUNCTICEPS','OPISTOGNATHUSAURIFRONS','PAGRUSPAGRUS','PARANTHIASFURCIFER','POMACANTHUSARCUATUS','POMACANTHUSPARU',
       'POMACANTHUS','POMACENTRIDAE','POMACENTRUSPARTITUS','POMACENTRUS','PRIACANTHUSARENATUS','PRISTIGENYSALTA','PRISTIPOMOIDESAQUILONARIS','PSEUDUPENEUSMACULATUS','PTEROIS',
       'RACHYCENTRONCANADUM','RHOMBOPLITESAURORUBENS','RYPTICUSMACULATUS','SCARIDAE','SCARUSVETULA','SERIOLADUMERILI','SERIOLAFASCIATA','SERIOLARIVOLIANA','SERIOLA',
       'SERIOLAZONATA','SERRANUSANNULARIS','SERRANUSPHOEBE','SERRANUS','SPARIDAE','SPARISOMAAUROFRENATUM','SPARISOMAVIRIDE','SPHYRAENABARRACUDA','SPHYRNALEWINI',
       #'STENOTOMUSCAPRINUS','SYACIUM','SYNODONTIDAE','THALASSOMABIFASCIATUM']
       'STENOTOMUSCAPRINUS','SYACIUM','SYNODONTIDAE','THALASSOMABIFASCIATUM','UPENEUSPARVUS','XANTHICHTHYSRINGENS']


#print('results on test image',results)
#print('shape of results',results.shape)

## Output CSV file path
#csv_file_path = 'detections_4_12.csv'
csv_file_path = 'detections_4_12n.csv'

# Header for the CSV file
csv_header = ['# 1: Detection or Track-id', '2: Video or Image Identifier', '3: Unique Frame Identifier',
              '4-7: Img-bbox(TL_x', 'TL_y', 'BR_x', 'BR_y)', '8: Detection or Length Confidence',
              '9: Target Length (0 or -1 if invalid)', '10-11+: Repeated Species', 'Confidence Pairs or Attributes']

# Save detections to CSV
csv_data = [csv_header]

#directory = r'/work/cshah/updatedYOLOv8/ultralytics/extracted_FRAMESn/'

#print('directory of extracted frames',directory)

output_fol = r'/work/cshah/updatedYOLOv8/ultralytics/extracted_frames_411/'

# Dictionary to map image names to unique frame identifiers

video_path =r'../../datasets/2021TestVideo/762101178_cam3.avi'
# Open the video file
cap = cv2.VideoCapture(video_path)

frame_rate = 5  # Frames per second

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()


image_path_saved = '/work/cshah/updatedYOLOv8/ultralytics/extracted_FRAMESn/'

frame_count = 0
frame_id_dict = {}

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Check if it's time to save a frame based on the frame rate
    if frame_count % int(cap.get(cv2.CAP_PROP_FPS) / frame_rate) == 0:
        ### Save the frame as a PNG image with a filename like '762101028_cam3_1.png'
        #frame_filename = os.path.join(output_folder, f'762101028_cam3_{frame_count // int(cap.get(cv2.CAP_PROP_FPS) / frame_rate) + 1}.png')

        #frame_filename = os.path.join(output_fol, f'762101178_cam3_{frame_count // int(cap.get(cv2.CAP_PROP_FPS) / frame_rate) + 1}.png')
        frame_filenamen = f'762101178_cam3_{frame_count // int(cap.get(cv2.CAP_PROP_FPS) / frame_rate) + 1}.png'

        img_id = frame_count // int(cap.get(cv2.CAP_PROP_FPS) / frame_rate) + 1
        print('image id',img_id)


        image_names = frame_filenamen
        ###print('frame filenames',frame_filename)
        #print('frame filenames new',frame_filenamen)

        print('image names',image_names) 

        #cv2.imwrite(frame_filename, frame)

        image_name = image_names
        # Get or create unique frame identifier
        unique_frame_id = frame_id_dict.get(image_name, len(frame_id_dict))
        frame_id_dict[image_name] = unique_frame_id
        print('unique_frame_id number',unique_frame_id)

        ###image_path = image_names
        #image_path = image_path_saved
        image_path = os.path.join(testset, f'{image_name}')
        detections = model(image_path)

    #frame_count += 1


        # Process and save the detection results
        detections_to_save = []

        print('length of detections',len(detections))
     
        for i in range(len(detections)):    
            j = 0
            for result in detections:
                print('conf score in bbox',result.boxes.conf)
                ###while result.boxes.conf >= 0.5:
                #while len(result.boxes.xyxy) != 0:
                
                #j = 0
                if len(result.boxes.xyxy) != 0:
                #if len(result.boxes.xyxy) != 0:

                      print('bboxes in result',result.boxes)
                      print('results inDetections',result)

                      #bbox_pt = result.boxes[].item()                    

                      bbox_xyxyn = result.boxes.xyxy
                      bboxo = bbox_xyxyn[j].cpu().numpy()
                      print('bbox j iter',bboxo)

                      bbox_xyxy = result.boxes.xyxy
                      bbox_conf = result.boxes.conf
                      bbox_cls = result.boxes.cls
                      class_names = class_names
                      class_indices = bbox_cls.int()
                      class_names_detected = [class_names[idx] for idx in class_indices]
                      #print('box xyxy',bbox_xyxy)

                      #pt = (detections[0, i, j, 1:5] * torch.Tensor([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])).cpu().numpy()


                      bbox_xyxy = bbox_xyxy.cpu().numpy()
                      print('bbox in numpy',bbox_xyxy)
            
                      ######pt = bbox_xyxy
                      #bbox = bbox_xyxy
                      bbox = bboxo
                      #bboxnum = bbox.numpy()
                      bboxconf = bbox_conf[j].cpu().numpy()
                      

                      #class_name = class_names_detected
                      #confidence = bbox_conf
                      #detections_to_save.append([j + 1, image_name, unique_frame_id, *bbox, confidence, -1, class_name, confidence])
                      ####j = j + 1

                      confidence = bboxconf
                      class_name = class_names_detected[j]
                      detections_to_save.append([j + 1, image_name, unique_frame_id, *bbox, confidence, -1, class_name, confidence])
  

                      print('box xyxy',bbox_xyxy)
                      ####print('box xywh',bbox_xywh)
                      print('conf pred',bbox_conf)
                      print('class no',bbox_cls)
                      print('class_names_detected',class_names_detected)

                      #j = j + 1

            j = j + 1

        csv_data.extend(detections_to_save)

    frame_count += 1


# Write CSV data to file
with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerows(csv_data)

print(f'Detections saved to {csv_file_path}')

    ##frame_count += 1



    #frame_count += 1

# Release the video capture object
cap.release()
