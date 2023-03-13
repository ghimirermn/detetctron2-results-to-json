## A simple python script to convert detectron2 predictions to json format. 

# **prediction_to_coco.py** converts detectron2 to json format compatible with coco format with predcition masks in polygon format.  


detectron2 produces coco_instances_results.json file for evaluation part where the json file contains class prediction, score prediction and segmentation mask prediction (in RLE format). However, one may want an automatically created **_coco formatted json_** file of a prediction image. 

Curently there are no scripts doing that, and this repo tries to fill that missing part.

Furthermore, unlike in RLE format, this code presents results in segmented masks in **polygon** format.

# **dd2_to_labelme_format.py** converts detectron2 detection predictions to json format compatible with labelme format.
