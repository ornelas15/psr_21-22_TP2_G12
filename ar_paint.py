#!/usr/bin/python3
import numpy as np
import cv2
import argparse

clicking = False
color = (0, 0, 255) # BGR
painting = np.ones((550, 700, 3)) * 255  # start with everything white


# obtain a numbered inverted image, labels and label-color matches
def load_coloring_image():
    cImage = cv2.imread("./images/ovni.png", cv2.IMREAD_GRAYSCALE)
    
    ret, thresh = cv2.threshold(cImage,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # use connectedComponentWithStats to find the white areas
    connectivity = 4
    output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)

    num_labels = output[0] # number of labels / areas
    labels = output[1] # label matrix
    stats = output[2] # statistics
    centroids = output[3] # centroid matrix
    
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
                    labelColors[labels[i][j]] = colors[labels[i][j]%3]
    
    # write the numbers on the image
    fontScale = (width * height) / (800 * 800)
    for i in range(0, len(centroids)):
        if labelColors[i] != (0,0,0):
            cv2.putText(cImage,str(i), (int(centroids[i][0]-fontScale*15), int(centroids[i][1]+fontScale*15)), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0,0,0), 1)

    cImage = cv2.bitwise_not(cImage)
    
    return cv2.cvtColor(cImage,cv2.COLOR_GRAY2RGB), labelColors, labels


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
    # -----------------------------------
    # Initialize
    # -----------------------------------
    global painting, color, labels
    
    # Define argparse inputs
    parser = argparse.ArgumentParser(description='Definition of test mode')
    parser.add_argument('-cim','--coloring_image_mode', action='store_true', help='If present, it will presented a coloring image to paint.', required=False)

    # Parse arguments
    args = parser.parse_args()
    
    window_name = "Augmented Reality Paint"
    print(window_name)
    
    # coloring image mode
    if args.coloring_image_mode:
        cImage, labelColors, labels = load_coloring_image()
        height, width, _ = cImage.shape
        painting = np.ones((height, width, 3)) * 255

        cv2.imshow(window_name, cv2.subtract(painting, cImage, dtype=cv2.CV_64F))
    else:
        cv2.imshow(window_name, painting)
        
    cv2.setMouseCallback(window_name, mouse_paint)

    # -----------------------------------
    # Execution
    # -----------------------------------
    while True:
        if args.coloring_image_mode:
            cv2.imshow(window_name, cv2.subtract(painting, cImage, dtype=cv2.CV_64F))
        else:
            cv2.imshow(window_name, painting)

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
            elif key == ord('Q') or key == ord('q') or key == 27: # 27 -> ESC
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
                    print(hits/(hits+misses))
                print("Quitting")
                break
    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
