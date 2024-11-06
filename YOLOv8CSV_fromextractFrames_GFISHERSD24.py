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
#model = YOLO('/work/cshah/YOLOv8_weights_saved/Yolov8m_enh_128batch/weights/best.pt')

##best weight for YOLOv8 on GFISHERSD24
#model = YOLO('/work/cshah/updatedYOLOv8/ultralytics/runs/detect/train88/weights/best.pt')
model = YOLO('/work/cshah/updatedYOLOv8/ultralytics/runs/detect/train176/weights/best.pt') ##77.6, 49.1




testset = '/work/cshah/updatedYOLOv8/ultralytics/extracted_FRAMESn/'

#print('name of testset in 1',testset[40])
#video_path = '../../datasets/2021TestVideo/762101449_cam3.avi'
#results = model('../../datasets/images/test//YSC4_Camera4_08-07-19_19-01-400004.png',save=True)


class_names= ['ACANTHURUS-170160100','ACANTHURUSCOERULEUS-170160102','ALECTISCILIARIS-170110101','ANISOTREMUSVIRGINICUS-170190105','ANOMURA-999100401','ARCHOSARGUSPROBATOCEPHALUS-170213601',
'BALISTESCAPRISCUS-189030502','BALISTESVETULA-189030504','BODIANUSPULCHELLUS-170280201','BODIANUSRUFUS-170280202','CALAMUS-170210600','CALAMUSBAJONADO-170210602','CALAMUSLEUCOSTEUS-170210604',
'CALAMUSNODOSUS-170210608','CALAMUSPRORIDENS-170210605','CALLIONYMIDAE-170420000','CANTHIDERMISSUFFLAMEN-189030402','CANTHIGASTERROSTRATA-189080101','CARANGIDAE-170110000',
'CARANXBARTHOLOMAEI-170110801','CARANXCRYSOS-170110803','CARANXRUBER-170110807','CARCHARHINUS-108020200','CARCHARHINUSFALCIFORMIS-108020202','CARCHARHINUSPEREZI-108020211',
'CARCHARHINUSPLUMBEUS-108020208','CAULOLATILUSCHRYSOPS-170070104','CAULOLATILUSCYANOPS-170070101','CAULOLATILUSMICROPS-170070103','CENTROPRISTISOCYURUS-170024804',
'CENTROPRISTISPHILADELPHICA-170024805','CEPHALOPHOLISCRUENTATA-170020401','CEPHALOPHOLISFULVA-170020403','CHAETODON-170260300','CHAETODONCAPISTRATUS-170260302',
'CHAETODONOCELLATUS-170260307','CHAETODONSEDENTARIUS-170260309','CHROMIS-170270300','CHROMISENCHRYSURUS-170270302','CHROMISINSOLATUS-170270304',
'DECAPTERUS-170111200','DERMATOLEPISINERMIS-170020301','DIODONTIDAE-189090000','DIPLECTRUM-170020900','DIPLECTRUMFORMOSUM-170020903','EPINEPHELUS-170021200',
'EPINEPHELUSADSCENSIONIS-170021203','EPINEPHELUSMORIO-170021211','EQUETUSLANCEOLATUS-170201104','GOBIIDAE-170550000','GONIOPLECTRUSHISPANUS-170021403',
'GYMNOTHORAXMORINGA-143060202','GYMNOTHORAXSAXICOLA-143060205','HAEMULONALBUM-170191002','HAEMULONAUROLINEATUM-170191003','HAEMULONFLAVOLINEATUM-170191005',
'HAEMULONMACROSTOMUM-170191017','HAEMULONMELANURUM-170191007','HAEMULONPLUMIERI-170191008','HALICHOERES-170281200','HALICHOERESBATHYPHILUS-170281201',
'HALICHOERESBIVITTATUS-170281202','HALICHOERESGARNOTI-170281205','HOLACANTHUS-170290100','HOLACANTHUSBERMUDENSIS-170290102','HOLOCENTRUS-161110200',
'HOLOCENTRUSADSCENSIONIS-161110201','HYPOPLECTRUS-170021500','HYPOPLECTRUSGEMMA-170021503','HYPOPLECTRUSUNICOLOR-170021501','HYPORTHODUSFLAVOLIMBATUS-170021206',
'HYPORTHODUSNIGRITUS-170021202','IOGLOSSUS -170550800','IOGLOSSUS-170550800','KYPHOSUS-170240300','LACHNOLAIMUSMAXIMUS-170281801','LACTOPHRYSTRIGONUS-189070205',
'LIOPROPOMAEUKRINES-170025602','LUTJANUS-170151100','LUTJANUSANALIS-170151101','LUTJANUSAPODUS-170151102','LUTJANUSBUCCANELLA-170151106','LUTJANUSCAMPECHANUS-170151107',
'LUTJANUSGRISEUS-170151109','LUTJANUSSYNAGRIS-170151113','LUTJANUSVIVANUS-170151114','MALACANTHUSPLUMIERI-170070301','MULLOIDICHTHYSMARTINICUS-170220101',
'MURAENARETIFERA-143060402','MYCTEROPERCA-170022100','MYCTEROPERCABONACI-170022101','MYCTEROPERCAINTERSTITIALIS-170022103','MYCTEROPERCAMICROLEPIS-170022104',
'MYCTEROPERCAPHENAX-170022105','OCYURUSCHRYSURUS-170151501','OPHICHTHUSPUNCTICEPS-143150402','OPHICHTHUSPUNCTICEPS-143150402','OPISTOGNATHUS-170310200',
'OPISTOGNATHUSAURIFRONS-170310203','PAGRUSPAGRUS-170212302','PARANTHIASFURCIFER-170022701','PAREQUESUMBROSUS-170201105','POMACANTHUS-170290200',
'POMACANTHUSARCUATUS-170290201','POMACANTHUSPARU-170290203','POMACENTRIDAE-170270000','POMACENTRUS-170270500','POMACENTRUSPARTITUS-170270502',
'PRIACANTHUSARENATUS-170050101','PRISTIGENYSALTA-170050401','PRISTIPOMOIDES-170151800','PROGNATHODESACULEATUS-170260305',
'PROGNATHODESAYA-170260301','PRONOTOGRAMMUSMARTINICENSIS-170025101','PSEUDUPENEUSMACULATUS-170220701','PTEROIS-168011900','RACHYCENTRONCANADUM-170100101',
'RHOMBOPLITESAURORUBENS-170152001','RYPTICUSMACULATUS-170030106','SCARIDAE-170300000','SCARUSVETULA-170301107','SCOMBEROMORUS-170440800',
'SERIOLA-170113100','SERIOLADUMERILI-170113101','SERIOLAFASCIATA-170113103','SERIOLARIVOLIANA-170113105','SERIOLAZONATA-170113106','SERRANUS-170024200',
'SERRANUSANNULARIS-170024201','SERRANUSATROBRANCHUS-170024202','SERRANUSPHOEBE-170024208','SPARIDAE-170210000','SPARISOMAAUROFRENATUM-170301201',
'SPARISOMAVIRIDE-170301206','SPHYRAENABARRACUDA-165030101','SPHYRNALEWINI-108040102','STENOTOMUSCAPRINUS-170213403','SYACIUM-183011000','SYNODONTIDAE-129040000',
'THALASSOMABIFASCIATUM-170282801','UNKNOWNFISH','UPENEUSPARVUS-170220605','UROPHYCISREGIA-148010105','XANTHICHTHYSRINGENS-189030101']




#print('results on test image',results)
#print('shape of results',results.shape)

## Output CSV file path
#csv_file_path = 'detections_4_12.csv'
#csv_file_path = 'detections_4_12n.csv'
csv_file_path = 'detections_GFISHERS24.csv'


# Header for the CSV file
csv_header = ['# 1: Detection or Track-id', '2: Video or Image Identifier', '3: Unique Frame Identifier',
              '4-7: Img-bbox(TL_x', 'TL_y', 'BR_x', 'BR_y)', '8: Detection or Length Confidence',
              '9: Target Length (0 or -1 if invalid)', '10-11+: Repeated Species', 'Confidence Pairs or Attributes']

# Save detections to CSV
csv_data = [csv_header]

directory = r'/work/cshah/updatedYOLOv8/ultralytics/extracted_FRAMESn/'

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


##image_path_saved = '/work/cshah/updatedYOLOv8/ultralytics/extracted_FRAMESn/'

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
     
        #for i in range(len(detections)):    

        for i, result in enumerate(detections):

            print('Detection or Track-id:', i+1)
            #print('conf score in bbox:', result.conf)

            j = 0

            for j, bbox in enumerate(result.boxes):
            #for result in detections:
                print('conf score in bbox',result.boxes.conf)
                ###while result.boxes.conf >= 0.5:
                #while len(result.boxes.xyxy) != 0:
                
                ##j = 0

                if len(bbox.xyxy) != 0:

                #if len(result.boxes.xyxy) != 0:

                ##if len(result.boxes.xyxy) != 0:

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
                      #detections_to_save.append([i + 1, image_name, unique_frame_id, *bbox, confidence, -1, class_name, confidence])
  

                      print('box xyxy',bbox_xyxy)
                      ####print('box xywh',bbox_xywh)
                      print('conf pred',bbox_conf)
                      print('class no',bbox_cls)
                      print('class_names_detected',class_names_detected)

                      j = j + 1

            #j = j + 1
            #i = i + 1

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
