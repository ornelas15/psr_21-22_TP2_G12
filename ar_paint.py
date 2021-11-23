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
from colorama import Fore,Style

# GLOBAL VARIABLES
# -----------------------------------------------------
clicking = False
color = (0, 0, 255)  # BGR, Red by default
thickness = 2


# Obtain a numbered inverted image, labels and label-color matches
def load_coloring_image(height, width):
    cImage = cv2.imread("./images/ovni.png", cv2.IMREAD_GRAYSCALE)
    
    cImage = cv2.resize(cImage,   (int(cImage.shape[1] * height/cImage.shape[0]), height))

    ret, thresh = cv2.threshold(cImage, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    #height, width = thresh.shape

    cImage = np.zeros((height, width)).astype(np.uint8)
    
    print(width)
    print(thresh.shape)
    
    cImage[:, int(width/2 - thresh.shape[1]/2):int(width/2 + thresh.shape[1]/2)] = thresh

    # Use connectedComponentWithStats to find the white areas
    connectivity = 4
    output = cv2.connectedComponentsWithStats(cImage, connectivity, cv2.CV_32S)

    num_labels = output[0]  # number of labels / areas
    labels = output[1]  # label matrix
    stats = output[2]  # statistics
    centroids = output[3]  # centroid matrix
    print(stats)
    print(centroids)
    print(labels)

    # Associate a label with a color
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    labelColors = [None] * num_labels

    
    for i in range(height):
        for j in range(width):
            if not labelColors[labels[i][j]]:
                if cImage[i][j] == 0:
                    labelColors[labels[i][j]] = (0, 0, 0)
                else:
                    labelColors[labels[i][j]] = colors[labels[i][j] % 3]

    # Write the numbers on the image
    fontScale = (width * height) / (800 * 800)
    for i in range(0, len(centroids)):
        if labelColors[i] != (0, 0, 0):
            cv2.putText(cImage, str(i), (int(centroids[i][0] - fontScale * 15), int(centroids[i][1] + fontScale * 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 0), 1)

    cImage = cv2.bitwise_not(cImage)

    return cv2.cvtColor(cImage, cv2.COLOR_GRAY2RGB), labelColors, labels


# Deal with mouse events
def mouse_paint(event, x, y, flags, params):
    global clicking, painting

    if event == cv2.EVENT_MOUSEMOVE and clicking:
        painting[y, x] = color

    elif event == cv2.EVENT_LBUTTONDOWN:
        clicking = True

    elif event == cv2.EVENT_LBUTTONUP:
        clicking = False

# Advanced functionality 3
# Draw Rectangle 
def rectangle(event, x, y, flags, params):
    global painting, color, clicking, x1, y1, x2, y2, thickness
    if event == cv2.EVENT_LBUTTONDOWN:
        clicking = True
        x1 = x   
        y1 = y
        x2 = x
        y2 = y
    elif event == cv2.EVENT_MOUSEMOVE and clicking:
        copy = painting.copy()
        x2 = x
        y2 = y
        cv2.rectangle(copy, (x1, y1), (x2, y2), color, thickness)
        cv2.imshow('Augmented Reality Paint', copy)
    elif event == cv2.EVENT_LBUTTONUP:
        clicking = False
        cv2.rectangle(painting, (x1, y1), (x, y), color, thickness)

# Draw Circle
def circle(event, x, y, flags, params):
    global painting, color, clicking, x1, y1, x2, y2, thickness
    if event == cv2.EVENT_LBUTTONDOWN:
        clicking = True
        x1 = x   
        y1 = y
        x2 = x
        y2 = y
    elif event == cv2.EVENT_MOUSEMOVE and clicking:
        copy = painting.copy()
        x2 = x
        y2 = y
        cv2.circle(copy,(int((x2+x1)/2), int((y2+y1)/2)), int(sqrt((pow(((x2-x1)/2),2))+ pow(((y2-y1)/2),2))), color, thickness)
        cv2.imshow('Augmented Reality Paint', copy)
    elif event == cv2.EVENT_LBUTTONUP:
            clicking = False
            cv2.circle(painting, (int((x1+x)/2), int((y1+y)/2)), int(sqrt((pow(((x1-x)/2),2))+ pow(((y1-y)/2),2))) , color, thickness)


def main():
    # -----------------------------------------------------
    # Initialize
    # -----------------------------------------------------
    global painting, color, labels

    # Define argparse inputs
    parser = argparse.ArgumentParser(description='Definition of test mode')
    parser.add_argument('-j', '--json', type=str, required=True, help='Full path to json file.')
    parser.add_argument('-cim', '--coloring_image_mode', action='store_true', help='If present, it will presented a '
                                                                                   'coloring image to paint.',
                        required=False)
    parser.add_argument('-usp', '--use_shake_prevention', action='store_true', help='If present, it will prevent big '
                                                                                    'lines to appear across the white'
                                                                                    ' board', required=False)
    parser.add_argument('-uvs', '--use_video_stream', required=False, help='If present, the video capture window is '
                                                                           'used to draw instead of the white board')
    parser.add_argument('-um', '--use_mouse', action='store_true', help='If present, the position of the mouse will '
                                                                        'be used to draw', required=False)

    # Parse arguments
    args = parser.parse_args()

    # Open and Load json File
    with open(args.json) as f:
        ranges = json.load(f)
    pprint(ranges)

    # Configure opencv windows
    window1_name = "Augmented Reality Paint"
    window2_name = "Video Capture"
    window3_name = "Mask"
    cv2.namedWindow(window1_name, cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow(window2_name, cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow(window3_name, cv2.WINDOW_KEEPRATIO)
    print(window1_name)

    # Setup video capture for webcam
    capture = cv2.VideoCapture(0)

    _, image_capture = capture.read()
    height, width, _ = image_capture.shape
    painting = np.ones((height, width, 3)) * 255
    cv2.imshow(window1_name, painting)
    
    # Coloring image mode
    if args.coloring_image_mode:
        cImage, labelColors, labelMatrix = load_coloring_image(height, width)
        cv2.imshow(window1_name, cv2.subtract(painting, cImage, dtype=cv2.CV_64F))        

    cv2.setMouseCallback(window1_name, mouse_paint)

    x_last = None
    y_last = None
    thickness = 2   

    line = np.zeros((height, width, 3))

    # -----------------------------------------------------
    # Execution
    # -----------------------------------------------------
    while True:

        if args.coloring_image_mode:
            cv2.imshow(window1_name, cv2.subtract(painting, cImage, dtype=cv2.CV_64F))
        else:
            # Show White Board same size as capture
            cv2.imshow(window1_name, painting)
            
        # Get an image from the camera
        _, image_capture = capture.read()
        image_capture = cv2.flip(image_capture, 1) # Flip image

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

        # Get Object Max Area Centroid
        max_area = 0
        max_area_Label = 0
        for i in range(num_labels):
            if i != 0 and max_area < stats[i, cv2.CC_STAT_AREA]:
                max_area = stats[i, cv2.CC_STAT_AREA]
                max_area_Label = i

        mask2 = cv2.inRange(labels, max_area_Label, max_area_Label)
        mask2 = mask2.astype(bool)
        image_capture[mask2] = (0, 255, 0)

        # Draw Line on White Board
        x = int(centroids[max_area_Label, 0])
        y = int(centroids[max_area_Label, 1])
        if x_last != None and y_last != None:
            if thickness > 0:
                if args.use_shake_prevention:
                    # Distance = ((X2 - X1)² + (Y2 - Y1)²)**(1/2)
                    dist = ((x - x_last)**2 + (y - y_last)**2)**(1/2)
                    if dist < 50:
                        cv2.line(painting, (x, y), (x_last, y_last), color, thickness, cv2.LINE_4)
                    else:
                        cv2.line(painting, (x, y), (x, y), color, thickness, cv2.LINE_4)
                else:
                    cv2.line(painting, (x, y), (x_last, y_last), color, thickness, cv2.LINE_4)
            else:
                print('The thickness value has reached is limit, try to increase it')
                

            
            if thickness > 0:
            # Draw in Video Capture Test
                cv2.line(line, (x, y), (x_last, y_last), color, thickness, cv2.LINE_4)
                line = line.astype(np.uint8)
            # exit(0)
                image_capture = cv2.add(image_capture, line)
            else:
                 print('The thickness value has reached is limit, try to increase it')
                 
        x_last = x
        y_last = y

        # Show Capture Window
        cv2.imshow(window2_name, image_capture)
        
        cv2.resizeWindow(window1_name, (width//3, height//3))
        cv2.resizeWindow(window2_name, (width//3, height//3))
        cv2.resizeWindow(window3_name, (width//3, height//3))

        # Deal with keyboard events
        key = cv2.waitKey(20)

        if key != -1:
            print("Pressed key: " + chr(key))
            if key == ord('R') or key == ord('r'):
                color = (0, 0, 255)
                print('Red color selected')
            elif key == ord('G') or key == ord('g'):
                color = (0, 255, 0)
                print('Green color selected')
            elif key == ord('B') or key == ord('b'):
                color = (255, 0, 0)
                print('Blue color selected')
            elif key == ord('W') or key == ord('w'):
                name = str(ctime(time()))
                cv2.imwrite(name + '.jpg', painting)
            elif key == ord('E') or key == ord('e'):
                _, image_capture = capture.read()
                height, width, _ = image_capture.shape
                painting = np.ones((height, width, 3)) * 255
                cv2.imshow(window1_name, painting)
            elif key == ord('+'):
                    thickness += 1
                    print('Increase thickness')
            elif key == ord('-') and thickness > 0:
                    thickness -= 1
                    print('Decrease thickness')
            elif key == ord('C') or key == ord('c'):
                cv2.setMouseCallback(window1_name, circle)
                if not cv2.EVENT_MOUSEMOVE:
                    copy = painting.copy()
                    cv2.circle(copy, (int((x2+x1)/2), int((y2+y1)/2)), int(sqrt((pow(((x2-x1)/2),2))+ pow(((y2-y1)/2),2))) , color, thickness)
                    cv2.imshow(window1_name,copy)
            elif key == ord('R') or key == ord('r'):
                cv2.setMouseCallback(window1_name,rectangle)
                if not cv2.EVENT_MOUSEMOVE:
                    copy = painting.copy()
                    cv2.rectangle(copy,(x1,y1),(x2,y2),color,thickness)
                    cv2.imshow(window1_name,copy)
            elif key == ord('S') or key == ord('s'):
                cv2.setMouseCallback(window1_name,rectangle)
            elif key == ord('Q') or key == ord('q') or key == 27:  # 27 -> ESC
                if args.coloring_image_mode:
                    hits = 0
                    misses = 0
                    for i in range(height):
                        for j in range(width):
                            rightColor = labelColors[labelMatrix[i,j]]
                            if rightColor == (0, 0, 0):
                                pass
                            elif np.array_equal(painting[i,j], rightColor):
                                hits += 1
                            else:
                                misses += 1

                    print(hits)
                    print(hits / (hits + misses))
                print("Quitting")
                break

    # -----------------------------------------------------
    # TERMINATION
    # -----------------------------------------------------
    cv2.destroyAllWindows()
    capture.release()


if __name__ == '__main__':
    main()
