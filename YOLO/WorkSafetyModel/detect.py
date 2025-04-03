import os
import re
import sys
import pkg_resources
import subprocess as sp
from ultralytics import YOLO
from colorama import Fore, Style

# Functions ----------------------------------------------
def check_dir():
    print('Checking present working directory...')
    if re.search('WorkSafetyModel$', os.getcwd()):
        # print(Fore.GREEN + 'You are in the correct directory.' + Style.RESET_ALL)
        pass
    else:
        print(Fore.YELLOW + 'You are in the wrong directory. Please change to the YOLO/WorkSafetyModel directory.' + Style.RESET_ALL)
        quit()

def check_libraries():
    required_libraries = ['opencv-python', 'colorama', 'ultralytics']
    print('Checking for required libraries...')
    installed_libraries = {pkg.key for pkg in pkg_resources.working_set}
    missing_libraries = [lib for lib in required_libraries if lib not in installed_libraries]
    if missing_libraries:
        print(Fore.YELLOW + f'You are missing the following libraries: {missing_libraries} please install them' + Style.RESET_ALL)
        quit()

def check_model():
    print('Checking for safety model...')
    try:
        model = YOLO('runs/detect/train/weights/best.pt')
        print(Fore.GREEN + 'Safety model found.' + Style.RESET_ALL)
    except FileNotFoundError as err:
        print(Fore.RED + str(err))
        print(Fore.RED + 'Please let me know and I will attempt to fix on your personal machine. This is a major error.' + Style.RESET_ALL)

def find_video():
    print('Checking for video analysis file...')
    fp = 'safety_video.py'
    if os.path.exists(fp):
        # print(Fore.GREEN + 'Video analysis file found.' + Style.RESET_ALL)
        pass
    else:
        print(Fore.YELLOW + 'Video analysis file not found. Check the YOLO/WorkSafetyModel directory.' + Style.RESET_ALL)

def initial_checks():
    print()
    check_dir()
    check_libraries()
    check_model()
    find_video()

# MAIN ---------------------------------------------------
# Checks prerequisites before running the safety_video.py script
initial_checks()

print('starting video analysis script with the Work Safety Model... Press \'Q\' at any time to stop analysis')
fp = 'safety_video.py'
#sp.call('python ' + fp, shell=True)
os.system(f'python {fp}')
