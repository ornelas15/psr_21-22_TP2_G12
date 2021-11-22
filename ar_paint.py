#!/usr/bin/python3

# IMPORTS
# -----------------------------------------------------
import json
from time import time, ctime
from pprint import pprint
import numpy as np
import cv2
import argparse
import copy

# GLOBAL VARIABLES
# -----------------------------------------------------
clicking = False
color = (0, 0, 255)  # BGR, Red by default


# obtain a numbered inverted image, labels and label-color matches
def load_coloring_image():
    cImage = cv2.imread("./images/ovni.png", cv2.IMREAD_GRAYSCALE)

    ret, thresh = cv2.threshold(cImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # use connectedComponentWithStats to find the white areas
    connectivity = 4
    output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)

    num_labels = output[0]  # number of labels / areas
    labels = output[1]  # label matrix
    stats = output[2]  # statistics
    centroids = output[3]  # centroid matrix

    # associate a label with a color
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    labelColors = dict()

    height, width = cImage.shape
    for i in range(height):
        for j in range(width):
            if labels[i][j] not in labelColors.keys():
                if thresh[i][j] == 0:
                    labelColors[labels[i][j]] = (0, 0, 0)
                else:
                    labelColors[labels[i][j]] = colors[labels[i][j] % 3]

    # write the numbers on the image
    fontScale = (width * height) / (800 * 800)
    for i in range(0, len(centroids)):
        if labelColors[i] != (0, 0, 0):
            cv2.putText(cImage, str(i), (int(centroids[i][0] - fontScale * 15), int(centroids[i][1] + fontScale * 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 0), 1)

    cImage = cv2.bitwise_not(cImage)

    return cv2.cvtColor(cImage, cv2.COLOR_GRAY2RGB), labelColors, labels


# deal with mouse events
def mouse_paint(event, x, y, flags, params):
    global clicking, painting

    if event == cv2.EVENT_MOUSEMOVE and clicking:
        painting[y, x] = color

    elif event == cv2.EVENT_LBUTTONDOWN:
        clicking = True

    elif event == cv2.EVENT_LBUTTONUP:
        clicking = False


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

    # configure opencv windows
    window1_name = "Augmented Reality Paint"
    window2_name = "Video Capture"
    window3_name = "Mask"
    print(window1_name)

    # setup video capture for webcam
    capture = cv2.VideoCapture(0)

    # coloring image mode
    if args.coloring_image_mode:
        cImage, labelColors, labels = load_coloring_image()
        height, width, _ = cImage.shape
        painting = np.ones((height, width, 3)) * 255
        cv2.imshow(window1_name, cv2.subtract(painting, cImage, dtype=cv2.CV_64F))
    else:
        _, image_capture = capture.read()
        height, width, _ = image_capture.shape
        painting = np.ones((height, width, 3)) * 255
        cv2.imshow(window1_name, painting)

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
            # get an image from the camera
            _, image_capture = capture.read()
            image_capture = cv2.flip(image_capture, 1) # Flip image

            # Show White Board same size as capture
            cv2.imshow(window1_name, painting)

            # Processing Mask
            mins = np.array([ranges['limits']['B']['min'], ranges['limits']['G']['min'], ranges['limits']['R']['min']])
            maxs = np.array([ranges['limits']['B']['max'], ranges['limits']['G']['max'], ranges['limits']['R']['max']])
            mask = cv2.inRange(image_capture, mins, maxs)
            mask = mask.astype(bool)  # conversion from numpy uint8 to bool
            image_processed = copy.deepcopy(image_capture)
            image_processed[np.logical_not(mask)] = 0

            # Show Mask Window
            cv2.imshow(window3_name, image_processed)

            # Get Object
            image_grey = cv2.cvtColor(image_processed, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(image_grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)

            # Get Object Max Area Centroid
            max_area = 0
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
                if args.use_shake_prevention:
                    # Distancia = ((X2 - X1)² + (Y2 - Y1)²)**(1/2)
                    dist = ((x - x_last)**2 + (y - y_last)**2)**(1/2)
                    if dist < 50:
                        cv2.line(painting, (x, y), (x_last, y_last), color, thickness, cv2.LINE_4)
                    else:
                        cv2.line(painting, (x, y), (x, y), color, thickness, cv2.LINE_4)
                else:
                    cv2.line(painting, (x, y), (x_last, y_last), color, thickness, cv2.LINE_4)

                # Draw in Video Capture Test
                cv2.line(line, (x, y), (x_last, y_last), color, thickness, cv2.LINE_4)
                print(line.dtype)
                line = line.astype(np.uint8)
                print(line.dtype)
                # exit(0)
                image_capture = cv2.add(image_capture, line)
            x_last = x
            y_last = y

            # Show Capture Window
            cv2.imshow(window2_name, image_capture)

        # deal with keyboard events
        key = cv2.waitKey(20)

        if key != -1:
            print("Tecla pressionada: " + str(key))
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
            elif key == ord('C') or key == ord('c'):
                _, image_capture = capture.read()
                height, width, _ = image_capture.shape
                painting = np.ones((height, width, 3)) * 255
                cv2.imshow(window1_name, painting)
            elif key == ord('+'):
                thickness += 1
                print('Incrise thickness')
            elif key == ord('-') and thickness > 0:
                thickness -= 1
                print('Decrise thickness')
            elif key == ord('Q') or key == ord('q') or key == 27:  # 27 -> ESC
                if args.coloring_image_mode:
                    hits = 0
                    misses = 0
                    for i in range(height):
                        for j in range(width):
                            rightColor = labelColors[labels[i][j]]
                            if np.array_equal(painting[i][j], rightColor):
                                hits += 1
                            elif rightColor != (0, 0, 0):
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
