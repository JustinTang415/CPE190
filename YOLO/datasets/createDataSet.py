from ultralytics import YOLO

model = YOLO('yolo11n.pt')

f = open('YOLO/datasets/coco.txt', 'a')
for object in model.names.values():
    f.write(object + '\n')
f.close()