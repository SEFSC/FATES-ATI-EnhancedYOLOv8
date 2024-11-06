from ultralytics import YOLO
import os
import cv2
import csv
import numpy as np
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Function to initialize the distributed environment
def init_distributed_mode():
    if not dist.is_available():
        raise RuntimeError("Distributed package is not available")
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank())

# Function for distributed processing
def run_detection(rank, world_size, csv_file_path):
    # Initialize distributed environment
    init_distributed_mode()

    # Check if CUDA is available and set the device
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    # Load YOLOv8 model
    model = YOLO('/work/cshah/updatedYOLOv8/ultralytics/runs/detect/train176/weights/best.pt')
    model.model.to(device)

    # Directory for test frames
    testset = '/work/cshah/updatedYOLOv8/ultralytics/extracted_FRAMESn/'

    # List of class names
    class_names = ['ACANTHURUS-170160100', 'ACANTHURUSCOERULEUS-170160102', 'ALECTISCILIARIS-170110101', ...]

    # Define video path and frame extraction settings
    video_path = r'../../datasets/2021TestVideo/762101178_cam3.avi'
    cap = cv2.VideoCapture(video_path)
    frame_rate = 5  # Extract frames every 5 seconds

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    frame_count = 0
    frame_id_dict = {}
    tracker = {}
    csv_data = []  # Initialize csv_data as an empty list

    def iou(box1, box2):
        """Compute the Intersection over Union (IoU) of two bounding boxes."""
        x1, y1, x2, y2 = box1
        x1p, y1p, x2p, y2p = box2
        
        # Compute intersection
        ix1 = max(x1, x1p)
        iy1 = max(y1, y1p)
        ix2 = min(x2, x2p)
        iy2 = min(y2, y2p)
        
        iw = max(ix2 - ix1 + 1, 0)
        ih = max(iy2 - iy1 + 1, 0)
        inter_area = iw * ih
        
        # Compute union
        box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
        box2_area = (x2p - x1p + 1) * (y2p - y1p + 1)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract frames at the specified rate
        if frame_count % int(cap.get(cv2.CAP_PROP_FPS) / frame_rate) == 0:
            frame_filenamen = f'762101178_cam3_{frame_count // int(cap.get(cv2.CAP_PROP_FPS) / frame_rate) + 1}.png'
            img_id = frame_count // int(cap.get(cv2.CAP_PROP_FPS) / frame_rate) + 1
            image_names = frame_filenamen

            image_name = image_names
            unique_frame_id = frame_id_dict.get(image_name, len(frame_id_dict))
            frame_id_dict[image_name] = unique_frame_id
            image_path = os.path.join(testset, f'{image_name}')

            # Run detection
            detections = model(image_path)

            detections_to_save = []

            # Process detections
            for i, result in enumerate(detections):
                for j, bbox in enumerate(result.boxes):
                    if bbox.conf >= 0.5 and len(bbox.xyxy) != 0:
                        bbox_xyxyn = result.boxes.xyxy
                        bboxo = bbox_xyxyn[j].cpu().numpy()
                        bbox_conf = result.boxes.conf
                        bbox_cls = result.boxes.cls
                        class_names_detected = [class_names[idx] for idx in bbox_cls.int()]
                        bboxconf = bbox_conf[j].cpu().numpy()
                        confidence = bboxconf
                        class_name = class_names_detected[j]

                        # Track detection
                        max_iou = 0
                        best_track_id = None
                        for track_id, (previous_bbox, _) in tracker.items():
                            iou_score = iou(bboxo, previous_bbox)
                            if iou_score > max_iou:
                                max_iou = iou_score
                                best_track_id = track_id
                        
                        if max_iou > 0.3:  # IoU threshold for tracking
                            tracker[best_track_id] = (bboxo, confidence)
                            track_id = best_track_id
                        else:
                            track_id = len(tracker) + 1
                            tracker[track_id] = (bboxo, confidence)

                        detections_to_save.append([track_id, image_name, unique_frame_id, *bboxo, confidence, -1, class_name, confidence])

            csv_data.extend(detections_to_save)

        frame_count += 1

    # Save detections to CSV
    csv_header = ['# 1: Detection or Track-id', '2: Video or Image Identifier', '3: Unique Frame Identifier',
                  '4-7: Img-bbox(TL_x', 'TL_y', 'BR_x', 'BR_y)', '8: Detection or Length Confidence',
                  '9: Target Length (0 or -1 if invalid)', '10-11+: Repeated Species', 'Confidence Pairs or Attributes']

    with open(csv_file_path, 'a', newline='') as csvfile:  # Append mode
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(csv_header)
        csv_writer.writerows(csv_data)

    print(f'Detections saved to {csv_file_path}')

    cap.release()

# Main function to start the distributed processing
def main():
    world_size = 4  # Number of GPUs
    csv_file_path = 'detection_output_distributed.csv'  # CSV file to save detections

    # Use multiprocessing to run distributed processes
    mp.spawn(run_detection, args=(world_size, csv_file_path), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
