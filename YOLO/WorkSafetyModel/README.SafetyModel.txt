All steps assume that ultralytics (and YOLOv8) was installed correctly

data.yaml has already been created @ YOLO\WorkSafetyModel\data.yaml

DO NOT FOLLOW THESE STEPS I HAVE ALREADY DONE ALL THIS FOR YOU. THESE STEPS HAVE BEEN WRITTEN OUT FOR REFERENCE.
IF YOU WANT TO TEST OUT THE WORK SAFETY MODEL; JUST RUN THE image_analysis.py FILE

**If image_analysis.py does not work, make sure you are in the correct directory. You should be in ..\CPE190\YOLO\WorkSafetyModel

TRAINING:
1. Modify data.yaml to contain FULL Paths (not relative) for the train, val and test images folders
2. run the following command to train the model (This step will take 4-8 hours):
    yolo task=detect mode=train model=yolov8n.yaml data=/path/to/data.yaml epochs=50 imgsz=640

TESTING:
1. You can evaluate the model using the following command:
    yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=/path/to/data.yaml imgsz=640
2. Initial testing can be done with this command:
    yolo task=detect mode=predict model=runs/detect/train/weights/best.pt source=/path/to/images_or_video imgsz=640
The previous command should give output an analyzed image; If no mishaps then we're all good