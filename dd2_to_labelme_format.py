### script to convert detectron2 detection output to json format (labelme)

import json
import os
import labelme
import base64
import cv2

def base64imgdata(img_path):
    """ returns image as base64 data, required for labelme format """
    data = labelme.LabelFile.load_image_file(img_path)
    image_data = base64.b64encode(data).decode('utf-8')
    return image_data

def instances_labelme_json(image_name,bbox, pred_class, shape_type):

    """
    Dump on "Instance" object to a labelme-format json.

    Args:
        bbox (bounding box):
        pred_class (prediction class):

    Returns:
         list[dict]: list of json annotations in labelme format.
    """

    img = cv2.imread(image_name)
    imageHeight, imageWidth, _ = img.shape
    imageData = base64imgdata(image_name)

    results = []
    N = 2
    for k in range(len(bbox)):
        bb = [bbox[k][n:n+N] for n in range(0, len(bbox[k]), N)] # making a sub-lists with 2 coordinates in each

        #conversion from coco to labelme format (x[0][0]->no change, x[0][1]->no change, x[1][0]= x[1][0] + x[0][0], x[1][1]= x[0][1] + x[1][1]

        bb[1][0] = bb[1][0] + bb[0][0]
        bb[1][1] = bb[0][1] + bb[1][1]

        result = {
            "label": str(pred_class[k]),
            "points": bb,
            "group_id": None,
            "shape_type": shape_type
                 }
        results.append(result)

    final = { "shapes": results, "imagePath": os.path.basename(image_name), "imageData": imageData, "imageHeight": imageHeight, "imageWidth": imageWidth }
    image_name = os.path.splitext(image_name)[0]
    with open(os.path.join("results", os.path.basename(image_name) + ".json"), "w") as fp:
        json.dump(final,fp, indent= 2)
