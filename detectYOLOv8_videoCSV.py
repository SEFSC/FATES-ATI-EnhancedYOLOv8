import cv2
import time
import torch
from ultralytics import YOLO
import os

#model = YOLO('/work/cshah/YOLOv8_weights_saved/YOLOvn/weights/best.pt')

model = YOLO('/work/cshah/YOLOv8_weights_saved/Yolov8m_enh_128batch/weights/best.pt')


### Load YOLOv5 model
#weights_path = 'path/to/your/yolov5/weights.pt'
#model = torch.hub.load('ultralytics/yolov5:v6.0', 'yolov5s', weights=weights_path).autoshape()

# Load video  
#video_path = 'path/to/your/video.mp4'

#video_path = '../../datasets/2021TestVideo/762101178_cam3.avi'
#video_path = '../../datasets/2021TestVideo/762101001_cam3.avi'
#video_path = '../../datasets/2021TestVideo/762101002_cam3.avi'
#video_path = '../../datasets/2021TestVideo/762101525_cam3.avi'
#video_path = '../../datasets/2021TestVideo/762101457_cam3.avi'
#video_path = '../../datasets/2021TestVideo/762101449_cam3.avi'

#video_path = '../../datasets/2021TestVideo/762101084_cam3.avi'
#video_path = '../../datasets/2021TestVideo/762101513_cam3.avi'  ##29 FPS
video_path = '../../datasets/2021TestVideo/762101515_cam3.avi' ##31 FPS for frame count 1


output_dir = '/work/cshah/updatedYOLOv8/ultralytics/extractedFRAMES'

cap = cv2.VideoCapture(video_path)

def count_frames(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video file is opened successfully
    if not cap.isOpened():
        print("Error: Could not open the video file")
        return -1
    
    # Get the total number of frames in the video
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Release the video capture object
    cap.release()
    
    return num_frames

# Count the number of frames in the video
num_frames = count_frames(video_path)

if num_frames != -1:
    print("Number of frames in the video:", num_frames)


# Initialize variables for FPS calculation
frame_count = 0
total_detection_time = 0

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
       ###'STENOTOMUSCAPRINUS','SYACIUM','SYNODONTIDAE','THALASSOMABIFASCIATUM']
       'STENOTOMUSCAPRINUS','SYACIUM','SYNODONTIDAE','THALASSOMABIFASCIATUM','UPENEUSPARVUS','XANTHICHTHYSRINGENS']


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform YOLOv5 detection on the frame
    start_time = time.time()
    results = model(frame)
    end_time = time.time()
    #print('results:',results)


for result in results:
    #detection results
    bbox_xyxy = result.boxes.xyxy
    bbox_xywh = result.boxes.xywh
    #bbox_xyxyn = result.boxes.xyxyn
    #bbox_xywhn = result.boxes.xywhn
    bbox_conf = result.boxes.conf
    bbox_cls = result.boxes.cls
    class_names = class_names
    class_indices = bbox_cls.int()
    class_names_detected = [class_names[idx] for idx in class_indices]

    print('box xyxy',bbox_xyxy)
    print('box xywh',bbox_xywh)
    print('conf pred',bbox_conf)
    print('class no',bbox_cls)
    print('class_names_detected',class_names_detected)


     

    # Example: Extract detection or track IDs
    detection_track_ids = [1, 2, 3, 4, 5, 6, 7]  # Example detection or track IDs
    
    # Print the Detection or Track IDs
    print("Detection or Track IDs:", detection_track_ids)

        
    # Construct the filename for the frame
    frame_name = f"762101513_cam3_{frame_count}"
    print('frame name',frame_name)

    ### Save the frame name to a text file
    with open(os.path.join(output_dir, 'frame_names.txt'), 'a') as f:
        f.write(frame_name + '\n')

    Unique_Frame_Identifier = frame_count
    print('Unique Frame Identifier',Unique_Frame_Identifier)
    ##Unique_Frame_Identifier = Unique_Frame_Identifier + 1

    #if detection_track_ids
    #Detection_or_Track_id =  

    ### Save the frame name to a text file
    with open(os.path.join(output_dir, 'frame_names.txt'), 'a') as f:
        f.write(frame_name + '\n')

    # Calculate detection time for the current frame
    detection_time = end_time - start_time

    # Accumulate total detection time
    total_detection_time += detection_time

    # Increment frame count
    frame_count += 1

    #Unique_Frame_Identifier = frame_count
    #print('Unique Frame Identifier',Unique_Frame_Identifier)

    # Display FPS every 100 frames
    if frame_count % 10 == 0: ##100 originally
        average_fps = frame_count / total_detection_time
        print(f'Average FPS: {average_fps:.2f}')

# Release video capture
cap.release()

# Calculate and print final average FPS
average_fps = frame_count / total_detection_time
print(f'Final Average FPS: {average_fps:.2f}')
