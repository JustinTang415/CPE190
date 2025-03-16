import cv2
from ultralytics import YOLO
from PIL import Image
import random

def resetFrame():
    width = 200
    height = 150

    img = Image.new('RGB', (width, height), color = 'white')
    img.save('YOLO/inference/images/frame.png')

# opening dataset in read mode and creating a list of objects in the dataset
my_f = open('YOLO/datasets/coco.txt', 'r')
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
model = YOLO('YOLO/weights/yolo11n.pt')

frame_width = 600
frame_height = 440

# Open the video file
#capture =  cv2.VideoCapture(0) # for webcam
capture = cv2.VideoCapture('YOLO/inference/videos/Devasated_Joey.mp4') # for video file

if not capture.isOpened():
    print('Error opening video stream or file')
    exit()

while True:
    ret, frame = capture.read()
    if not ret:
        print('Can\'t receive frame (stream end?). Exiting ...')
        break
    
    # Resize frame
    frame = cv2.resize(frame, (frame_width, frame_height))

    # write the frame
    cv2.imwrite('YOLO/inference/images/frame.png', frame)

    # Perform inference (DO NOT set save to True; it will save every frame as an image in the inference folder)
    detect_params = model.predict(source='YOLO/inference/images/frame.png', conf=0.45, project='YOLO/inference/predictions', save=False)

    # convert tensor array to numpy
    detect_params = detect_params[0].numpy()

    if len(detect_params) != 0:
        # the target object of the current frame
        obj_num = 0

        # loop through the detected objects
        for param in detect_params.boxes.xyxy.tolist():

            # draw a BBox around detected objects
            cv2.rectangle(frame, (int(param[0]), int(param[1])), (int(param[2]), int(param[3])), detection_colors[int(detect_params.boxes.cls[obj_num])], 3)
            
            # label the detected objects
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(frame, class_list[int(detect_params.boxes.cls[obj_num])] + " " + str(round(detect_params.boxes.conf[obj_num], 3) * 100) + "%", (int(param[0]), int(param[1])-10), font, 1, (255, 255, 255),2)
            obj_num += 1

    # Display the resulting frame
    cv2.imshow('Object Detection', frame)

    # Press Q on keyboard to exit
    if cv2.waitKey(1) == ord('q'):
        print('Escape key pressed; Exiting ...')
        break

# When everything done, release the video capture object
resetFrame()
capture.release()
cv2.destroyAllWindows()
