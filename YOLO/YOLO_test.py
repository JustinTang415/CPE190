from ultralytics import YOLO

# load a pretrained YOLOv8 model
model = YOLO('yolo11n.pt')

# predict on an image
detection_output = model.predict(source="YOLO/inference/images/img0.jpg", conf=0.25, project='YOLO/inference/predictions', save=True)

# display tensor array
print(detection_output)

# display numpy array
print(detection_output[0].numpy())
