from ultralytics import YOLO
from colorama import Fore, Style

try:
    # load the work safety model
    model = YOLO('runs/detect/train/weights/best.pt')

    # predict on an image
    detection_output = model.predict(source="inference/images/img1.jpg", conf=0.25, project='inference/predictions', save=True)

except FileNotFoundError as err:
    print(Fore.RED + str(err))
    print(Fore.YELLOW + 'You are in the wrong directory. Please change to the YOLO/WorkSafetyModel directory.' + + Style.RESET_ALL)
    quit()

# display tensor array
#print(detection_output)

# display numpy array
print(detection_output[0].numpy())
