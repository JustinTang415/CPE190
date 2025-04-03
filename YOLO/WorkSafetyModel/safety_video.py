import cv2
from ultralytics import YOLO
from PIL import Image
from colorama import Fore, Style
import random

# Functions ----------------------------------------------
def resetFrame():
    width = 200
    height = 150

    img = Image.new('RGB', (width, height), color = 'white')
    img.save('inference/images/frame.png')

def get_frame_size(cap):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * 0.5
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * 0.5
    return int(width), int(height)

# MAIN ---------------------------------------------------
try:
    # opening dataset in read mode and creating a list of objects in the dataset
    my_f = open('datasets/safetyParameters.txt', 'r')
    data = my_f.read()
    class_list = data.split('\n')
    my_f.close()

except FileNotFoundError as err:
    print(Fore.RED + str(err))
    print(Fore.YELLOW + 'You are in the wrong directory. Please change to the YOLO/WorkSafetyModel directory.' + Style.RESET_ALL)
    quit()

# print(class_list)

# Generate random colors for each class
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))

# load Safety Model
model = YOLO('runs/detect/train/weights/best.pt')

# Open the video file
#capture =  cv2.VideoCapture(0) # for webcam
capture = cv2.VideoCapture('inference/videos/construction_Broll.mp4') # for video file

if not capture.isOpened():
    print('Error opening video stream or file')
    exit()

frame_width, frame_height = get_frame_size(capture)

no_detection_streak = 0

while True:
    ret, frame = capture.read()
    if not ret:
        print('Can\'t receive frame (stream end?). Exiting ...')
        break
    
    # Resize frame
    frame = cv2.resize(frame, (frame_width, frame_height))

    # write the frame
    cv2.imwrite('inference/images/frame.png', frame)

    # Perform inference (DO NOT set save to True; it will save every frame as an image in the inference folder)
    detect_params = model.predict(source='inference/images/frame.png', conf=0.45, project='inference/predictions', save=False, verbose=False)

    # convert tensor array to numpy
    detect_params = detect_params[0].numpy()

    if len(detect_params) != 0:
        # the target object of the current frame
        obj_num = 0

        # loop through the detected objects
        for param in detect_params.boxes.xyxy.tolist():

            # draw a BBox around detected object
            cv2.rectangle(frame, (int(param[0]), int(param[1])), (int(param[2]), int(param[3])), detection_colors[int(detect_params.boxes.cls[obj_num])], 3)
            
            # label the BBox with the object name and confidence
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(frame, class_list[int(detect_params.boxes.cls[obj_num])] + " " + str(round(detect_params.boxes.conf[obj_num], 3) * 100) + "%", (int(param[0]), int(param[1])-10), font, 1, (255, 255, 255),2)
            obj_num += 1

    # Display the resulting frame
    cv2.imshow('Object Detection', frame)

    # count number of detected people and hardhats
    people = detect_params.boxes.cls.tolist().count(float(4))
    hardhats = detect_params.boxes.cls.tolist().count(float(0))

    # Check if people are wearing hardhats, and keep track of how many frames in a row this has been true
    if hardhats != people:
        no_hat_person_detected = abs(people - hardhats)
        no_detection_streak += 1
    else:
        no_detection_streak = 0
    
    # If the number of frames without detection exceeds a certain threshold, send an alert
    # change the no_detection_streak upper bound as needed
    if no_detection_streak >= 30 and no_hat_person_detected > 0:
        no_hat_person_detected = 0
        print(Fore.RED + 'ALERT: No hardhat detected on person!' + Style.RESET_ALL)
        print('Saving last frame with no detection ...')
        cv2.imshow('detected violation', frame) # TODO: Save the last frame with no detection and send it to the user (this probably won't work)

    # Press Q on keyboard to exit
    if cv2.waitKey(1) == ord('q'):
        print('Escape key pressed; Exiting ...')
        break

# When everything done, release the video capture object
resetFrame()
capture.release()
cv2.destroyAllWindows()
