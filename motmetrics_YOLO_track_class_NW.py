import motmetrics as mm
## List all default metrics
#mh = mm.metrics.create()
#print(mh.list_metrics_markdown())

### Load ground truth and YOLO predictions
#gt = mm.io.loadtxt('groundtruth.txt', fmt='mot15-2D')
#pred = mm.io.loadtxt('predicted.txt', fmt='mot15-2D')

# Define the paths to the ground truth and predicted tracking files
gt_txt_path = '/work/cshah/updatedYOLOv8/ultralytics/groundtruthMOT.txt'
pred_txt_path = '/work/cshah/updatedYOLOv8/ultralytics/track_results_yolo_mot_.txt'

# Load ground truth and YOLO predictions
gt = mm.io.loadtxt(gt_txt_path, fmt='mot15-2D')
pred = mm.io.loadtxt(pred_txt_path, fmt='mot15-2D')


# Create an accumulator for each class
accumulators = {}

# Iterate through unique classes in the ground truth or predictions
for cls in set(gt['class_id']).union(pred['class_id']):
    accumulators[cls] = mm.MOTAccumulator(auto_id=True)

    # Filter data by class
    gt_cls = gt[gt['class_id'] == cls]
    pred_cls = pred[pred['class_id'] == cls]

    # Process each frame
    for frame in range(1, max(gt_cls.index.max(), pred_cls.index.max()) + 1):
        gt_dets = gt_cls.loc[frame].values[:, :4] if frame in gt_cls.index else []
        pred_dets = pred_cls.loc[frame].values[:, :4] if frame in pred_cls.index else []
        dists = mm.distances.iou_matrix(gt_dets, pred_dets, max_iou=0.5)
        accumulators[cls].update(
            gt_cls.loc[frame].index if frame in gt_cls.index else [],
            pred_cls.loc[frame].index if frame in pred_cls.index else [],
            dists
        )

# Compute metrics for each class
mh = mm.metrics.create()
for cls, acc in accumulators.items():
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name=f'class_{cls}')
    print(f"Metrics for class {cls}:\n", summary)

# Print available metrics
print(mh.list_metrics_markdown())
