#!/usr/bin/python3

# IMPORTS
# -----------------------------------------------------
import json
from time import time, ctime
from pprint import pprint
import numpy as np
import cv2
import argparse
from copy import deepcopy
from math import sqrt, pow
from colorama import Fore, Style
import random

# GLOBAL VARIABLES
# -----------------------------------------------------
clicking = False
x, y = 0, 0


# Obtain a numbered inverted image, labels and label-color matches
def load_coloring_image(height, width):
    cImage = cv2.imread(["./images/ovni.png", "./images/other.png"][random.randint(0,1)], cv2.IMREAD_GRAYSCALE)

    cImage = cv2.resize(cImage, (int(cImage.shape[1] * height / cImage.shape[0]), height))

    ret, thresh = cv2.threshold(cImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cImage = np.zeros((height, width)).astype(np.uint8)

    cImage[:, int(width / 2 - thresh.shape[1] / 2):int(width / 2 + thresh.shape[1] / 2)] = thresh

    # Use connectedComponentWithStats to find the white areas
    connectivity = 4
    output = cv2.connectedComponentsWithStats(cImage, connectivity, cv2.CV_32S)

    num_labels = output[0]  # number of labels / areas
    labels = output[1]  # label matrix
    stats = output[2]  # statistics
    centroids = output[3]  # centroid matrix

    # Associate a label with a color
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    labelColors = [None] * num_labels

    for i in range(height):
        for j in range(width):
            if not labelColors[labels[i][j]]:
                if cImage[i][j] == 0:
                    labelColors[labels[i][j]] = (0, 0, 0)
                else:
                    labelColors[labels[i][j]] = colors[random.randint(0,2)]

    # Write the numbers on the image
    fontScale = (width * height) / (800 * 800) / 2
    for i in range(0, len(centroids)):
        if labelColors[i] != (0, 0, 0):
            cv2.putText(cImage, str(i), (int(centroids[i][0] - fontScale * 13), int(centroids[i][1] + fontScale * 8)),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, (0, 0, 0), 1)

    cImage = cv2.bitwise_not(cImage)

    return cv2.cvtColor(cImage, cv2.COLOR_GRAY2RGB), labelColors, labels


# Deal with mouse events
def mouse_paint(event, mx, my, flags, params):
    global clicking, x, y

    if event == cv2.EVENT_MOUSEMOVE and clicking:
        x = mx
        y = my

    elif event == cv2.EVENT_LBUTTONDOWN:
        clicking = True

    elif event == cv2.EVENT_LBUTTONUP:
        clicking = False


def main():
    # -----------------------------------------------------
    # Initialize
    # -----------------------------------------------------
    global clicking, x, y

    # Define argparse inputs
    parser = argparse.ArgumentParser(description='Definition of test mode')
    parser.add_argument('-j', '--json', type=str, required=True, help='Full path to json file.')
    parser.add_argument('-cim', '--coloring_image_mode', action='store_true', help='If present, it will presented a '
                                                                                   'coloring image to paint.',
                        required=False)
    parser.add_argument('-usp', '--use_shake_prevention', action='store_true', help='If present, it will prevent big '
                                                                                    'lines to appear across the white'
                                                                                    ' board', required=False)
    parser.add_argument('-uvs', '--use_video_stream', action='store_true', required=False, help='If present, the '
                                                                                                'video capture window '
                                                                                                'is '
                                                                                                'used to draw instead of the white board')

    # Parse arguments
    args = parser.parse_args()

    # Open and Load json File
    with open(args.json) as f:
        ranges = json.load(f)

    # Configure opencv windows
    window1_name = "Augmented Reality Paint"
    window2_name = "Video Capture"
    window3_name = "Mask"

    cv2.namedWindow(window1_name, cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow(window2_name, cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow(window3_name, cv2.WINDOW_KEEPRATIO)
    print('\n' + Fore.GREEN + Style.BRIGHT + window1_name + Style.RESET_ALL)
    print(Fore.CYAN + Style.BRIGHT + " \n Test functionalities:" + Style.RESET_ALL)
    print(Fore.CYAN + "  Red line color (press R)" + Style.RESET_ALL)
    print(Fore.CYAN + "  Green line color (press G)" + Style.RESET_ALL)
    print(Fore.CYAN + "  Blue line color (press B)" + Style.RESET_ALL)
    print(Fore.CYAN + "  Increase line thickness (press +)" + Style.RESET_ALL)
    print(Fore.CYAN + "  Decrease line thickness (press -)" + Style.RESET_ALL)
    print(Fore.CYAN + "  Image Capture (press E)" + Style.RESET_ALL)
    print(Fore.CYAN + "  Draw Rectangle (press S)" + Style.RESET_ALL)
    print(Fore.CYAN + "  Draw Circle (press C)" + Style.RESET_ALL)
    print(Fore.CYAN + "  Terminate program (press Q)" + Style.RESET_ALL)

    # Setup video capture for webcam
    capture = cv2.VideoCapture(0)

    _, image_capture = capture.read()
    height, width, _ = image_capture.shape

    # painting = np.ones((height, width, 3)) * 255

    if args.use_video_stream:
        painting = np.zeros((height, width, 3))
        cv2.imshow(window1_name, image_capture)
    else:
        painting = np.ones((height, width, 3)) * 255
        cv2.imshow(window1_name, painting)

    cv2.setMouseCallback(window1_name, mouse_paint)

    # Coloring image mode
    if args.coloring_image_mode:
        cImage, labelColors, labelMatrix = load_coloring_image(height, width)
        cv2.imshow(window1_name, cv2.subtract(painting, cImage, dtype=cv2.CV_64F))
        
        redLine="RED: "
        blueLine="BLUE: "
        greenLine="GREEN: "
        for i in range(len(labelColors)):
            if labelColors[i] == (0,0,255):
                redLine += str(i) + ", "
            elif labelColors[i] == (0,255,0):
                greenLine += str(i) + ", "
            elif labelColors[i] == (255,0,0):
                blueLine += str(i) + ", "
        print(Style.BRIGHT + "Colors to fill on the coloring image:" + Style.RESET_ALL)
        print(Fore.RED + Style.BRIGHT + redLine[:-2] + Style.RESET_ALL)
        print(Fore.GREEN + Style.BRIGHT + greenLine[:-2] + Style.RESET_ALL)
        print(Fore.BLUE + Style.BRIGHT + blueLine[:-2] + Style.RESET_ALL)

    x_last = None
    y_last = None

    x1, y1, x2, y2 = 0, 0, 0, 0

    thickness = 2

    drawing = False

    color = (0, 0, 255)  # BGR, Red by default

    # -----------------------------------------------------
    # Execution
    # -----------------------------------------------------
    while True:

        if args.coloring_image_mode:
            cv2.imshow(window1_name, cv2.subtract(painting, cImage, dtype=cv2.CV_64F))
        elif args.use_video_stream:
            cv2.imshow(window1_name, image_capture)
        else:
            cv2.imshow(window1_name, painting)

        # Get an image from the camera
        _, image_capture = capture.read()
        image_capture = cv2.flip(image_capture, 1)  # Flip image

        # Processing Mask
        mins = np.array([ranges['limits']['B']['min'], ranges['limits']['G']['min'], ranges['limits']['R']['min']])
        maxs = np.array([ranges['limits']['B']['max'], ranges['limits']['G']['max'], ranges['limits']['R']['max']])
        mask = cv2.inRange(image_capture, mins, maxs)
        mask = mask.astype(bool)  # conversion from numpy uint8 to bool
        image_processed = deepcopy(image_capture)
        image_processed[np.logical_not(mask)] = 0

        # Show Mask Window
        cv2.imshow(window3_name, image_processed)

        # Get Object
        image_grey = cv2.cvtColor(image_processed, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(image_grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)

        # only update coordinates if it in fact finds 2 components
        if num_labels > 1:
            # Get Object Max Area Centroid
            max_area_Label = \
                sorted([(i, stats[i, cv2.CC_STAT_AREA]) for i in range(num_labels)], key=lambda x: x[1])[-2][0]

            mask2 = cv2.inRange(labels, max_area_Label, max_area_Label)
            mask2 = mask2.astype(bool)
            image_capture[mask2] = (0, 255, 0)

            # Cross On Centroid
            cv2.line(image_capture, (x, y), (x + 5, y), (0, 0, 255), 2, cv2.LINE_4)
            cv2.line(image_capture, (x, y), (x - 5, y), (0, 0, 255), 2, cv2.LINE_4)
            cv2.line(image_capture, (x, y), (x, y + 5), (0, 0, 255), 2, cv2.LINE_4)
            cv2.line(image_capture, (x, y), (x, y - 5), (0, 0, 255), 2, cv2.LINE_4)

            # if mouse not pressed use centroid coordinates
            if not clicking:
                x = int(centroids[max_area_Label, 0])
                y = int(centroids[max_area_Label, 1])

        # Show Capture Window
        cv2.imshow(window2_name, image_capture)

        # Draw Line on Painting
        if not drawing and x_last != None and y_last != None:
            if args.use_shake_prevention:
                # Distance = ((X2 - X1)² + (Y2 - Y1)²)**(1/2)
                dist = ((x - x_last) ** 2 + (y - y_last) ** 2) ** (1 / 2)
                if dist < 50:
                    cv2.line(painting, (x, y), (x_last, y_last), color, thickness, cv2.LINE_4)
                else:
                    cv2.line(painting, (x, y), (x, y), color, thickness, cv2.LINE_4)
            else:
                cv2.line(painting, (x, y), (x_last, y_last), color, thickness, cv2.LINE_4)

        x_last = x
        y_last = y

        # make a copy of painting to join all "layers" of the image
        copy = painting.copy()

        if args.use_video_stream:
            # Draw in Video Capture Test
            painting_mask = deepcopy(copy)
            painting_mask = cv2.cvtColor(painting_mask.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            _, painting_mask = cv2.threshold(painting_mask, 0, 255, cv2.THRESH_BINARY)
            painting_mask = painting_mask.astype(bool)
            image_capture[painting_mask] = (0, 0, 0)
            copy = cv2.add(image_capture.astype(np.uint8), copy.astype(np.uint8))

        if drawing:
            y2 = y
            x2 = x

            if drawing == "S":
                cv2.rectangle(copy, (x1, y1), (x2, y2), color, thickness)
            elif drawing == "C":
                cv2.circle(copy, (int((x2 + x1) / 2), int((y2 + y1) / 2)),
                           int(sqrt((pow(((x2 - x1) / 2), 2)) + pow(((y2 - y1) / 2), 2))), color, thickness)

        if args.coloring_image_mode:
            copy = cv2.subtract(copy, cImage, dtype=cv2.CV_64F)

        cv2.imshow(window1_name, copy)

        # Show smaller windows
        cv2.resizeWindow(window1_name, (width // 3, height // 3))
        cv2.resizeWindow(window2_name, (width // 3, height // 3))
        cv2.resizeWindow(window3_name, (width // 3, height // 3))

        # Deal with keyboard events
        key = cv2.waitKey(20)

        if key != -1:
            print(Fore.CYAN + Style.BRIGHT + "\nPressed key: " + chr(key) + Style.RESET_ALL)
            if key == ord('R') or key == ord('r'):
                color = (0, 0, 255)
                print(Fore.RED + Style.BRIGHT + 'Red color selected' + Style.RESET_ALL)
            elif key == ord('G') or key == ord('g'):
                color = (0, 255, 0)
                print(Fore.GREEN + Style.BRIGHT + 'Green color selected' + Style.RESET_ALL)
            elif key == ord('B') or key == ord('b'):
                color = (255, 0, 0)
                print(Fore.BLUE + Style.BRIGHT + 'Blue color selected' + Style.RESET_ALL)
            elif key == ord('W') or key == ord('w'):
                name = str(ctime(time()))
                cv2.imwrite(name + '.jpg', painting)
                print(Fore.WHITE + Style.BRIGHT + 'Image saved' + Style.RESET_ALL)
            elif key == ord('C') or key == ord('c'):
                if args.use_video_stream:
                    painting = np.zeros((height, width, 3))
                else:
                    painting = np.ones((height, width, 3)) * 255
                print(Fore.WHITE + Style.BRIGHT + 'Image cleared' + Style.RESET_ALL)
            elif key == ord('+'):
                if thickness < 20:
                    thickness += 1
                    print(Fore.WHITE + Style.BRIGHT + 'Increase thickness' + Style.RESET_ALL)
                else:
                    print(
                        Fore.RED + Style.BRIGHT + 'The thickness value has reached is limit, try to decrease it' + Style.RESET_ALL)
            elif key == ord('-'):
                if thickness > 1:
                    thickness -= 1
                    print(Fore.WHITE + Style.BRIGHT + 'Decrease thickness' + Style.RESET_ALL)
                else:
                    print(
                        Fore.RED + Style.BRIGHT + 'The thickness value has reached is limit, try to increase it' + Style.RESET_ALL)
            elif key == ord('O') or key == ord('o'):
                if not drawing:
                    drawing = "C"
                    x1 = x
                    y1 = y
                    print(Fore.MAGENTA + Style.BRIGHT + 'Draw Circle Selected' + Style.RESET_ALL)
                else:
                    cv2.circle(painting, (int((x1 + x) / 2), int((y1 + y) / 2)),
                               int(sqrt((pow(((x1 - x) / 2), 2)) + pow(((y1 - y) / 2), 2))), color, thickness)
                    drawing = False
                    x1, y1, x2, y2 = 0, 0, 0, 0
                    print(Fore.MAGENTA + Style.BRIGHT + 'Circle Drawn' + Style.RESET_ALL)
            elif key == ord('S') or key == ord('s'):
                if not drawing:
                    drawing = "S"
                    x1 = x
                    y1 = y
                    print(Fore.MAGENTA + Style.BRIGHT + 'Draw Rectangle Selected' + Style.RESET_ALL)
                else:
                    cv2.rectangle(painting, (x1, y1), (x, y), color, thickness)
                    drawing = False
                    x1, y1, x2, y2 = 0, 0, 0, 0
                    print(Fore.MAGENTA + Style.BRIGHT + 'Rectangle Drawn' + Style.RESET_ALL)
            elif key == ord('Q') or key == ord('q') or key == 27:  # 27 -> ESC
                if args.coloring_image_mode:
                    hits = 0
                    misses = 0
                    for i in range(height):
                        for j in range(width):
                            rightColor = labelColors[labelMatrix[i, j]]
                            if rightColor == (0, 0, 0):
                                pass
                            elif np.array_equal(painting[i, j], rightColor):
                                hits += 1
                            else:
                                misses += 1

                    print(hits)
                    print(hits / (hits + misses))
                print('\n' + Fore.RED + Style.BRIGHT + "Quitting" + Style.RESET_ALL + '\n')
                break

    # -----------------------------------------------------
    # TERMINATION
    # -----------------------------------------------------
    cv2.destroyAllWindows()
    capture.release()


if __name__ == '__main__':
    main()
