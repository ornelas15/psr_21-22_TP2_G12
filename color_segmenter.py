#!/usr/bin/python3

# -------------------------------------------------------------------------------
# Name:        Color Segmentor
# Purpose:     Change the maximum and minimum limits for R, B and G and save the values
#
# Author:      Grupo 12
#
# Created:     18/11/2021
# -------------------------------------------------------------------------------


# IMPORTS
# -----------------------------------------------------
import copy
import json
import cv2
import numpy as np
from colorama import Fore, Style, Back


# FUNCTIONS
# -----------------------------------------------------
def onTrackbar(value):
    pass


def main():
    # -----------------------------------------------------
    # INITIALIZATION
    # -----------------------------------------------------

    # ranges dict
    ranges = {'limits': {'B': {'max': 255, 'min': 0},
                         'G': {'max': 255, 'min': 0},
                         'R': {'max': 255, 'min': 229}}}

    # setup video capture for webcam
    capture = cv2.VideoCapture(0)

    # configure opencv window
    window_name = 'Color Segmentor'
    cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)

    # Create Trackbar
    cv2.createTrackbar('MinB', window_name, 0, 256, onTrackbar)
    cv2.createTrackbar('MaxB', window_name, 256, 256, onTrackbar)
    cv2.createTrackbar('MinG', window_name, 0, 256, onTrackbar)
    cv2.createTrackbar('MaxG', window_name, 256, 256, onTrackbar)
    cv2.createTrackbar('MinR', window_name, 0, 256, onTrackbar)
    cv2.createTrackbar('MaxR', window_name, 256, 256, onTrackbar)

    # -----------------------------------------------------
    # EXECUTION
    # -----------------------------------------------------
    while True:
        # get an image from the camera
        _, image = capture.read()

        if image is None:
            print(Fore.CYAN + 'Video is over!' + Style.RESET_ALL)
            break  # video is over

        # Get Trackbar Position and update corresponding value in ranges dict
        ranges['limits']['B']['min'] = cv2.getTrackbarPos('MinB', window_name)
        ranges['limits']['B']['max'] = cv2.getTrackbarPos('MaxB', window_name)
        ranges['limits']['G']['min'] = cv2.getTrackbarPos('MinG', window_name)
        ranges['limits']['G']['max'] = cv2.getTrackbarPos('MaxG', window_name)
        ranges['limits']['R']['min'] = cv2.getTrackbarPos('MinR', window_name)
        ranges['limits']['R']['max'] = cv2.getTrackbarPos('MaxR', window_name)

        # Processing
        mins = np.array([ranges['limits']['B']['min'], ranges['limits']['G']['min'], ranges['limits']['R']['min']])
        maxs = np.array([ranges['limits']['B']['max'], ranges['limits']['G']['max'], ranges['limits']['R']['max']])
        mask = cv2.inRange(image, mins, maxs)
        mask = mask.astype(bool)  # conversion from numpy from uint8 to bool

        image_processed = copy.deepcopy(image)
        image_processed[np.logical_not(mask)] = 0

        height, width, _ = image_processed.shape

        # show window
        cv2.imshow(window_name, image_processed)

        cv2.resizeWindow(window_name, (width // 2, height // 2))

        # Read Key
        key = cv2.waitKey(30)

        if key == ord('w'):
            file_name = 'limits.json'
            with open(file_name, 'w') as file_handle:
                json.dump(ranges, file_handle)
        elif key == ord('q'):
            print(Fore.CYAN + '\nYou pressed q!\n' + Style.RESET_ALL)
            print(Fore.RED + Style.BRIGHT + Back.YELLOW + 'Terminating...' + Style.RESET_ALL)
            print('\n')
            break

    # -----------------------------------------------------
    # TERMINATION
    # -----------------------------------------------------
    cv2.destroyAllWindows()
    capture.release()

if __name__ == '__main__':
    main()
