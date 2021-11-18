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

    # setup video capture for video file
    # capture = cv2.VideoCapture('/home/guilherme/PSR_Repository/PSR/A6/Ex3/test2.mp4')

    # configure opencv window
    window_name = 'Color Segmentor'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

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
            print('Video is over!')
            break  # video is over

        # Get Trackbar Position
        MinB = cv2.getTrackbarPos('MinB', window_name)
        MaxB = cv2.getTrackbarPos('MaxB', window_name)
        MinG = cv2.getTrackbarPos('MinG', window_name)
        MaxG = cv2.getTrackbarPos('MaxG', window_name)
        MinR = cv2.getTrackbarPos('MinR', window_name)
        MaxR = cv2.getTrackbarPos('MaxR', window_name)

        # Update corresponding value in ranges dict
        ranges['limits']['B']['min'] = MinB
        ranges['limits']['B']['max'] = MaxB
        ranges['limits']['G']['min'] = MinG
        ranges['limits']['G']['max'] = MaxG
        ranges['limits']['R']['min'] = MinR
        ranges['limits']['R']['max'] = MaxR

        # Processing
        mins = np.array([ranges['limits']['B']['min'], ranges['limits']['G']['min'], ranges['limits']['R']['min']])
        maxs = np.array([ranges['limits']['B']['max'], ranges['limits']['G']['max'], ranges['limits']['R']['max']])
        mask = cv2.inRange(image, mins, maxs)
        mask = mask.astype(bool)  # conversion from numpy from uint8 to bool

        image_processed = copy.deepcopy(image)
        image_processed[np.logical_not(mask)] = 0

        # show window
        cv2.imshow(window_name, image_processed)

        # Read Key
        key = cv2.waitKey(30)

        if key == ord('w'):
            file_name = 'limits.json'
            with open(file_name, 'w') as file_handle:
                json.dump(ranges, file_handle)
        elif key == ord('q'):
            print('You pressed q! Terminating...')
            break

    # -----------------------------------------------------
    # TERMINATION
    # -----------------------------------------------------
    cv2.destroyAllWindows()
    capture.release()

if __name__ == '__main__':
    main()
