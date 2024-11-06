from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2


#model = YOLO("best.pt")
#cap =cv2.VideoCapture("fish.avi")

model = YOLO("/work/cshah/updatedYOLOv8/ultralytics/runs/detect/train176/weights/best.pt")
cap =cv2.VideoCapture("/work/cshah/unseen_videos_jackonedrive/2022 Pisces Video/762201029_cam3.avi")

#model = YOLO("best.pt")
#cap =cv2.VideoCapture("fish.avi")
assert cap.isOpened(), "Error reading video file"

# Get the dimensions of the video frame
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the region points based on the video dimensions
# Here, we're using the entire video frame, but this can be adjusted
expanded_region_points = [
    (0, 0),                              # Top-left corner
    (frame_width, 0),                     # Top-right corner
    (frame_width, frame_height),          # Bottom-right corner
    (0, frame_height)                     # Bottom-left corner
]
# Video Writer
video_writer = cv2.VideoWriter("ultralytics_object_counting.avi",
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                int(cap.get(5)),
                                (int(cap.get(3)), int(cap.get(4))))

## List of class names
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

# Pass the list to the ObjectCounter constructor
counter = object_counter.ObjectCounter(classes_names=class_names)

#counter = object_counter.ObjectCounter()  # Init Object Counter
##region_points = [(20, 400), (1500, 20), (1500, 1200), (820, 1080)]


counter.set_args(view_img=True,
                 reg_pts=expanded_region_points,
                 classes_names=model.names,
                 draw_tracks=True)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break
    tracks = model.track(im0, persist=True, show=False)
    im0 = counter.start_counting(im0, tracks)

    # Access and display in/out counts
    in_count = getattr(counter, 'in_counts', 0)
    out_count = getattr(counter, 'out_counts', 0)

    cv2.putText(im0, f"Total In : {in_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(im0, f"Total Out: {out_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 0, 255), 2, cv2.LINE_AA)

    video_writer.write(im0)

video_writer.release()
"""while cap.isOpened():
    success, frame = cap.read()
    if not success:
        exit(0)
    tracks = model.track(frame, persist=True, show=False)
    frame = counter.start_counting(frame, tracks)
    #video_writer.write(frame)

#video_writer.release()"""
