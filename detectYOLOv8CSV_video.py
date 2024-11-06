from ultralytics import YOLO
import pprint

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

results = model('../../datasets/2021TestVideo/762101449_cam3.avi',save=True)

#video_path = '../../datasets/2021TestVideo/762101449_cam3.avi'

#results = model('../../datasets/images/test//YSC4_Camera4_08-07-19_19-01-400004.png',save=True)


#print('pprint results to detect position')
#pprint.pprint(results)

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

### Extract bounding boxes and class names
### Extract bounding boxes and class names

#print('shape of results',results.shape)

for r in results:
    boxes = r.boxes
    for box in boxes:
        b = box.xyxy[0]   # get box coordinates
        confc = box.conf
        c = box.cls
    print('box',b)
    print('conf pred',confc)
    print('class',c)

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

    print('box xyxy',bbox_xyxy) ##Pascal VOC format
    ##print('box xywh',bbox_xywh)
    print('conf pred',bbox_conf)
    print('class no',bbox_cls)
    print('class_names_detected',class_names_detected)


# Extract bounding boxes, class indices, and confidence scores
#bounding_boxes = [result['bbox'] for result in results]
#class_indices = [result['class_idx'] for result in results]
#confidence_scores = [result['confidence'] for result in results]

#bounding_boxes = results[:, :4]
#class_indices = results[:, 4].astype(int)
#confidence_scores = results[:, 5]

#print('bounding_boxes pred on test results',bounding_boxes)

#bounding_boxes = [result['boxes'] for result in results]
#print('bounding boxes yolov8',bounding_boxes)

#confidence_scores = [result['confidence'] for result in results]
#print('confidence scores yolov8',confidence_scores)

#results = model('../../datasets/2021TestVideo/762101178_cam3.avi',save=True)


#result_folder = '/work/cshah/updatedYOLOv8/ultralytics/detectedresults'

##path = model.export(format="onnx")  # export the model to ONNX format
##print('path')
