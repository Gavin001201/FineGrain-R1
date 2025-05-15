from coco_evaluation import CocoDetectionEvaluator

evaluator = CocoDetectionEvaluator('/home/data/wyy/datasets/coco2017/annotations/instances_val2017.json')
### mAP and AP for all categories
results, per_class_results = evaluator.evaluate('/home/data/wyy/projects/Visual-RFT/eval_results/coco_eval/prediction_results_base.json', '/home/data/wyy/projects/Visual-RFT/eval_results/coco_eval/results')

### mAP and AP for selected categories
selected_cate = ['bus', 'train', 'fire hydrant', 'stop sign', 'cat', 'dog', 'bed', 'toilet']
# selected_cate = ['mouse', 'fork', 'hot dog', 'cat', 'airplane', 'suitcase', 'parking meter', 'sandwich', 'train', 'hair drier', 'toilet', 'toaster', 'snowboard', 'frisbee', 'bear']

AP_sum = 0
for item in per_class_results:
    for key, value in item.items():
        if key in selected_cate:
            print(f"Key: {key}, Value: {value}")
            AP_sum += value
print("mAP for selected categories: ", (AP_sum)/(len(selected_cate)))