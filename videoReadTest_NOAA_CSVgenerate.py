from google.cloud import storage
from ultralytics import YOLO
import os
import cv2
import csv
import torch
import tempfile

# Check if CUDA is available and set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load YOLOv8 model
model = YOLO('/work/cshah/updatedYOLOv8/ultralytics/runs/detect/train176/weights/best.pt')
model.model.to(device)

# Google Cloud Storage details
bucket_name = 'nmfs_odp_sefsc'
prefix = 'PEMD/VIDEO_DATA/GOM_REEF_FISH/SoJo_2022/'

### Local directories
#frame_dir = '/work/cshah/extracted_frames_yolo_19Pisces'
#csv_output_dir = '/work/cshah/generated_csv_yolo/2019_Pisces_Video'
frame_dir = '/work/cshah/extracted_frames_yolo_19Pisces'
csv_output_dir = '/work/cshah/generated_csv_yolo_cloud/2019_Pisces_Video_cloud'


# Ensure output directories exist
os.makedirs(frame_dir, exist_ok=True)
os.makedirs(csv_output_dir, exist_ok=True)

# List of class names
class_names = ['ACANTHURUS-170160100', 'ACANTHURUSCOERULEUS-170160102', 'ALECTISCILIARIS-170110101', 
                'ANISOTREMUSVIRGINICUS-170190105', 'ANOMURA-999100401', 'ARCHOSARGUSPROBATOCEPHALUS-170213601',
                'BALISTESCAPRISCUS-189030502', 'BALISTESVETULA-189030504', 'BODIANUSPULCHELLUS-170280201', 
                'BODIANUSRUFUS-170280202', 'CALAMUS-170210600', 'CALAMUSBAJONADO-170210602', 
                'CALAMUSLEUCOSTEUS-170210604', 'CALAMUSNODOSUS-170210608', 'CALAMUSPRORIDENS-170210605', 
                'CALLIONYMIDAE-170420000', 'CANTHIDERMISSUFFLAMEN-189030402', 'CANTHIGASTERROSTRATA-189080101', 
                'CARANGIDAE-170110000', 'CARANXBARTHOLOMAEI-170110801', 'CARANXCRYSOS-170110803', 
                'CARANXRUBER-170110807', 'CARCHARHINUS-108020200', 'CARCHARHINUSFALCIFORMIS-108020202', 
                'CARCHARHINUSPEREZI-108020211', 'CARCHARHINUSPLUMBEUS-108020208', 'CAULOLATILUSCHRYSOPS-170070104', 
                'CAULOLATILUSCYANOPS-170070101', 'CAULOLATILUSMICROPS-170070103', 'CENTROPRISTISOCYURUS-170024804', 
                'CENTROPRISTISPHILADELPHICA-170024805', 'CEPHALOPHOLISCRUENTATA-170020401', 
                'CEPHALOPHOLISFULVA-170020403', 'CHAETODON-170260300', 'CHAETODONCAPISTRATUS-170260302', 
                'CHAETODONOCELLATUS-170260307', 'CHAETODONSEDENTARIUS-170260309', 'CHROMIS-170270300', 
                'CHROMISENCHRYSURUS-170270302', 'CHROMISINSOLATUS-170270304', 'DECAPTERUS-170111200', 
                'DERMATOLEPISINERMIS-170020301', 'DIODONTIDAE-189090000', 'DIPLECTRUM-170020900', 
                'DIPLECTRUMFORMOSUM-170020903', 'EPINEPHELUS-170021200', 'EPINEPHELUSADSCENSIONIS-170021203', 
                'EPINEPHELUSMORIO-170021211', 'EQUETUSLANCEOLATUS-170201104', 'GOBIIDAE-170550000', 
                'GONIOPLECTRUSHISPANUS-170021403', 'GYMNOTHORAXMORINGA-143060202', 'GYMNOTHORAXSAXICOLA-143060205', 
                'HAEMULONALBUM-170191002', 'HAEMULONAUROLINEATUM-170191003', 'HAEMULONFLAVOLINEATUM-170191005', 
                'HAEMULONMACROSTOMUM-170191017', 'HAEMULONMELANURUM-170191007', 'HAEMULONPLUMIERI-170191008', 
                'HALICHOERES-170281200', 'HALICHOERESBATHYPHILUS-170281201', 'HALICHOERESBIVITTATUS-170281202', 
                'HALICHOERESGARNOTI-170281205', 'HOLACANTHUS-170290100', 'HOLACANTHUSBERMUDENSIS-170290102', 
                'HOLOCENTRUS-161110200', 'HOLOCENTRUSADSCENSIONIS-161110201', 'HYPOPLECTRUS-170021500', 
                'HYPOPLECTRUSGEMMA-170021503', 'HYPOPLECTRUSUNICOLOR-170021501', 'HYPORTHODUSFLAVOLIMBATUS-170021206', 
                'HYPORTHODUSNIGRITUS-170021202', 'IOGLOSSUS-170550800', 'KYPHOSUS-170240300', 
                'LACHNOLAIMUSMAXIMUS-170281801', 'LACTOPHRYSTRIGONUS-189070205', 'LIOPROPOMAEUKRINES-170025602', 
                'LUTJANUS-170151100', 'LUTJANUSANALIS-170151101', 'LUTJANUSAPODUS-170151102', 
                'LUTJANUSBUCCANELLA-170151106', 'LUTJANUSCAMPECHANUS-170151107', 'LUTJANUSGRISEUS-170151109', 
                'LUTJANUSSYNAGRIS-170151113', 'LUTJANUSVIVANUS-170151114', 'MALACANTHUSPLUMIERI-170070301', 
                'MULLOIDICHTHYSMARTINICUS-170220101', 'MURAENARETIFERA-143060402', 'MYCTEROPERCA-170022100', 
                'MYCTEROPERCABONACI-170022101', 'MYCTEROPERCAINTERSTITIALIS-170022103', 
                'MYCTEROPERCAMICROLEPIS-170022104', 'MYCTEROPERCAPHENAX-170022105', 
                'OCYURUSCHRYSURUS-170151501', 'OPHICHTHUSPUNCTICEPS-143150402', 'OPISTOGNATHUS-170310200', 
                'OPISTOGNATHUSAURIFRONS-170310203', 'PAGRUSPAGRUS-170212302', 'PARANTHIASFURCIFER-170022701', 
                'PAREQUESUMBROSUS-170201105', 'POMACANTHUS-170290200', 'POMACANTHUSARCUATUS-170290201', 
                'POMACANTHUSPARU-170290203', 'POMACENTRIDAE-170270000', 'POMACENTRUS-170270500', 
                'POMACENTRUSPARTITUS-170270502', 'PRIACANTHUSARENATUS-170050101', 'PRISTIGENYSALTA-170050401', 
                'PRISTIPOMOIDES-170151800', 'PROGNATHODESACULEATUS-170260305', 'PROGNATHODESAYA-170260301', 
                'PRONOTOGRAMMUSMARTINICENSIS-170025101', 'PSEUDUPENEUSMACULATUS-170220701', 'PTEROIS-168011900', 
                'RACHYCENTRONCANADUM-170100101', 'RHOMBOPLITESAURORUBENS-170152001', 'RYPTICUSMACULATUS-170030106', 
                'SCARIDAE-170300000', 'SCARUSVETULA-170301107', 'SCOMBEROMORUS-170440800', 'SERIOLA-170113100', 
                'SERIOLADUMERILI-170113101', 'SERIOLAFASCIATA-170113103', 'SERIOLARIVOLIANA-170113105', 
                'SERIOLAZONATA-170113106', 'SERRANUS-170024200', 'SERRANUSANNULARIS-170024201', 
                'SERRANUSATROBRANCHUS-170024202', 'SERRANUSPHOEBE-170024208', 'SPARIDAE-170210000', 
                'SPARISOMAAUROFRENATUM-170301201', 'SPARISOMAVIRIDE-170301206', 
                'SPHYRAENABARRACUDA-165030101', 'SPHYRNALEWINI-108040102', 'STENOTOMUSCAPRINUS-170213403', 
                'SYACIUM-183011000', 'SYNODONTIDAE-129040000', 'THALASSOMABIFASCIATUM-170282801', 
                'UNKNOWNFISH', 'UPENEUSPARVUS-170220605', 'UROPHYCISREGIA-148010105', 
                'XANTHICHTHYSRINGENS-189030101']

# Frame extraction settings
frame_rate = 5  # Extract frames every 5 seconds

def list_files_from_bucket(bucket_name, prefix):
    """List the files in bucket_name containing the specified prefix."""
    client = storage.Client.create_anonymous_client()
    files = [blob.name for blob in client.list_blobs(bucket_name, prefix=prefix)]
    return files

def download_file_from_bucket(bucket_name, blob_name):
    """Download a file from Google Cloud Storage to a temporary location."""
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_name = temp_file.name
        blob.download_to_filename(temp_file_name)
        return temp_file_name

def process_video(file_path, video_file):
    """Process the video file and count frames."""
    cap = cv2.VideoCapture(file_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_file}.")
        return

    frame_count = 0
    csv_data = []  # Initialize csv_data as an empty list

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % int(cap.get(cv2.CAP_PROP_FPS) / frame_rate) == 0:
            frame_filename = f"{os.path.splitext(video_file)[0]}_frame_{frame_count // int(cap.get(cv2.CAP_PROP_FPS) / frame_rate) + 1}.png"
            img_id = frame_count // int(cap.get(cv2.CAP_PROP_FPS) / frame_rate) + 1
            image_path = os.path.join(frame_dir, frame_filename)

            cv2.imwrite(image_path, frame)

            # Run detection
            results = model(image_path)

            # Process detections
            for result in results:
                for i, bbox in enumerate(result.boxes):
                    if bbox.conf >= 0.5 and len(bbox.xyxy) != 0:
                        bbox_xyxyn = result.boxes.xyxy
                        bboxo = bbox_xyxyn[i].cpu().numpy()
                        bbox_conf = result.boxes.conf
                        bbox_cls = result.boxes.cls

                        bboxconf = bbox_conf[i].cpu().numpy()
                        confidence = bboxconf

                        # Ensure that class index is within the range of class_names list
                        class_idx = int(bbox_cls[i].cpu().numpy())
                        if class_idx >= len(class_names):
                            print(f"Error: Detected class index {class_idx} is out of bounds for class_names list.")
                            continue  # Skip this detection

                        class_name = class_names[class_idx]

                        track_id = i + 1  # Assign a unique track ID for each detection in the frame

                        csv_data.append([track_id, frame_filename, img_id, *bboxo, confidence, -1, class_name, confidence])

        frame_count += 1

    cap.release()
    
    return csv_data

def main():
    files = list_files_from_bucket(bucket_name, prefix)
    
    for file_name in files:
        if file_name.endswith(('.avi', '.mp4', '.mov', '.mkv')):  # Filter for video files
            print(f"Processing file: {file_name}")
            temp_file = download_file_from_bucket(bucket_name, file_name)
            csv_data = process_video(temp_file, os.path.basename(file_name))
            os.remove(temp_file)  # Clean up temporary file
            
            # Save detections to CSV
            csv_file_path = os.path.join(csv_output_dir, f"{os.path.splitext(os.path.basename(file_name))[0]}_detections.csv")
            csv_header = ['# 1: Detection or Track-id', '2: Video or Image Identifier', '3: Unique Frame Identifier',
                          '4-7: Img-bbox(TL_x', 'TL_y', 'BR_x', 'BR_y)', '8: Detection or Length Confidence',
                          '9: Target Length (0 or -1 if invalid)', '10-11+: Repeated Species', 'Confidence Pairs or Attributes']

            with open(csv_file_path, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(csv_header)
                csv_writer.writerows(csv_data)

            print(f'Detections saved to {csv_file_path}')

            # Clean up extracted frames
            for frame_file in os.listdir(frame_dir):
                frame_path = os.path.join(frame_dir, frame_file)
                os.remove(frame_path)

    print("All videos processed.")

if __name__ == "__main__":
    main()
