import numpy as np
import cv2
from ultralytics import YOLO
import random

# opening dataset in read mode and creating a list of objects in the dataset
my_f = open('datasets/coco.txt', 'r')
data = my_f.read()
class_list = data.split('\n')
my_f.close()

# print(class_list)

# Generate random colors for each class
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))

# load YOLO
model = YOLO('weights/yolov8n.pt', 'v8')

frame_width = 640
frame_height = 480

capture =  cv2.VideoCapture(0)
#capture = cv2.VideoCapture('inference/videos/[insert video here].mp4')

if not capture.isOpened():
    print('Error opening video stream or file')
    exit()

while True:
    ret, frame = capture.read()
    if not ret:
        print('Can\'t receive frame (stream end?). Exiting ...')
        break
    
    # Resize frame
    # frame = cv2.resize(frame, (frame_width, frame_height))\

    # write the frame
    cv2.imwrite('inference/images/frame.png', frame)

    # Perform inference
    detect_params = model.predict(source='inference/images/frame.png', conf=0.45, save=False)

    # convert tensor array to numpy
    #print(detect_params[0].numpy())
    detect_params = detect_params[0].numpy()

    if len(detect_params) != 0:

        # loop through the detected objects
        for param in detect_params:
            print('Printing param[0]')
            param = param.orig_img[0]
            print(param[0])
            print('Printed param[0]')
            exit()

            # draw a box around detected objects and label them
            cv2.rectangle(frame, (int(param[0]), int(param[1])), (int(param[2]), int(param[3])), detection_colors[int(param[5])], 3)
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(frame, class_list[int(param[5])]+" "+str(round(param[4], 3)) + "%", (int(param[0]), int(param[1])-10), font, 1, (255, 255, 255),2)

    # Display the resulting frame
    cv2.imshow('Object Detection', frame)

    # Press Q on keyboard to exit
    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the video capture object
capture.release()
cv2.destroyAllWindows()
