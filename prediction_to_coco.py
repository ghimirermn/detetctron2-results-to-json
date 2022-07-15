import json
import cv2
from detectron2.structures import BoxMode
from pred_masks_polygon import binary_mask_to_polygon

def instances_coco_json(image_name,bbox, scores, pred_class, pred_masks):
    """
    Dump on "Instance" object to a coco-format json with polygon coordinates.

    Args:
        image_name (image name of the predicted image)
        bbox (bounding box):
        scores (prediction score):
        pred_class (prediction class):
        pred_masks (segmentated coordinates of predicted region):

    Returns:
         list[dict]: list o f json annotations in COCO format.
    """
    area=[]
    for i in range(len(bbox)):
        area.append(bbox[i][2]*bbox[i][3])

    num_instances = len(pred_masks)  #just to get length for the loop
    if num_instances == 0:
        return []


    images = [{"file_name": image_name, "id": 0}]
    supercategory = [ ] # make your own
    id = [ ] #make your own
    name = [ ] #make your own

    res = []
    for l in range(len(id)):
        result = {
            "supercategory": supercategory[l],
            "id" : id[l],
            "name": name[l],
        }
        res.append(result)

    results = []
    for k in range(len(bbox)):
        result = {
            "segmentation": pred_masks[k],
            "area": area[k],
            "bbox": bbox[k],
            "scores": scores[k],
            "image_id": 0,
            "category_id": pred_class[k],
            "id": k,
            "iscrowd" : 0,

        }
        results.append(result)

    final = {"images": images,"categories": res, "annotations": results}

    with open(image_name + ".json", "w") as fp:
        json.dump(final,fp)

if __name__ == "__main__":

    im = cv2.imread("image.jpg")
    outputs = predictor(im)

    masks = (outputs["instances"].pred_masks).to("cpu")
    masks = (1 * masks)

    pred_masks = []
    for i in range(len(outputs["instances"].pred_classes)):
        pred_masks.append(binary_mask_to_polygon(masks[i]))

    bbox = outputs["instances"].pred_boxes.to("cpu")
    bbox = bbox.tensor.numpy()
    bbox = BoxMode.convert(bbox, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    bbox = bbox.tolist()
    scores = outputs["instances"].scores.tolist()
    pred_class = outputs["instances"].pred_classes.tolist()

    instances_coco_json(image_name, bbox=bbox, scores=scores, pred_class=pred_class, pred_masks=pred_masks)