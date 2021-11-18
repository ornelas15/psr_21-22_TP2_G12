#!/usr/bin/python3
import numpy as np
import cv2

clicking = False
color = (0, 0, 255) # BGR
painting = np.ones((565, 718, 3)) * 255  # start with everything white


def mouse_paint(event, x, y, flags, params):
    global clicking, painting
    
    if event == cv2.EVENT_MOUSEMOVE and clicking:
        painting[y, x] = color

    elif event == cv2.EVENT_LBUTTONDOWN:
        clicking = True

    elif event == cv2.EVENT_LBUTTONUP:
        clicking = False


# funcionalidade avancada 4
def load_coloring_image():
    cImage = cv2.imread( "./images/ovni.png", cv2.IMREAD_GRAYSCALE)
    
    _, cImage = cv2.threshold(cImage, 128, 255, cv2.THRESH_BINARY_INV)
    
    height, width = cImage.shape
    
    areas = set()
    for i in range(0, height, 20):
        for j in range(0, width, 20):
            if cImage[i][j] == 0:
                image2 = cImage.copy().astype("uint8")

                mask = np.zeros((height+2, width+2), np.uint8)
                try:
                    cv2.floodFill(image2, mask, (j,i), 255, 10, 10)
                except:
                    print(i)
                    print(j)
                
                image2 = cv2.bitwise_not(image2)
                areas.add(tuple(image2.reshape((height*width))))
    
    print(len(areas))
    cv2.imshow('Image after Flood Fill with seed 430x30', cImage | image2)


def main():
    # -----------------------------------
    # Initialize
    # -----------------------------------
    global painting, color
    
    window_name = "Augmented Reality Paint"
    print(window_name)

    # funcionalidade avancada 4
    #image = load_coloring_image()
    
    cv2.imshow(window_name, painting)
    painting = painting.astype(np.uint8)

    cv2.setMouseCallback(window_name, mouse_paint)

    # -----------------------------------
    # Execution
    # -----------------------------------
    while True:
        cv2.imshow(window_name, painting)

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
                print("Quitting")
                break
    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
