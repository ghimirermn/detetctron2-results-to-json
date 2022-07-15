## A simple python script to convert detectron2 prediction to json_coco format with prediction masks in polygon format.


detectron2 produces coco_instances_results.json file for evaluation part where the json file contains class prediction, score prediction and segmentation mask prediction (in RLE format). However, one may want an automatically created **_coco formatted json_** file of a prediction image. Curently there are no scripts doing that, and this repo tries to fill that missing part.

Furthermore, unlike in RLE format, this code presents results in segmented masks in **polygon** format.

The code is fairly simple and takes motivations from existing codes. You can modify parts based on your need.
