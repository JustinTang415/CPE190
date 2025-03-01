from ultralytics import YOLO

model = YOLO('yolov8n.pt') # yolov3-v7

f = open('datasets/coco.txt', 'a')
for object in model.names.values():
    f.write(object + '\n')
f.close()