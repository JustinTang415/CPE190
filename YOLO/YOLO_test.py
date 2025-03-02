from ultralytics import YOLO

# load a pretrained YOLOv8 model
model = YOLO('yolov8n.pt', 'v8')

# predict on an image
detection_output = model.predict(source="YOLO/inference/images/img0.jpg", conf=0.25, project='YOLO', save=True)

# display tensor array
print(detection_output)

# display numpy array
print(detection_output[0].numpy())
