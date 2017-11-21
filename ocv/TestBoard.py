import cv2
import numpy as np

import cv2

def start():
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)
    vc.set(3,800)
    vc.set(4,800)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        rval, frame = vc.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        cv2.imshow("preview", gray)

        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break

    cv2.destroyWindow("preview")
    vc.release()

start()

# def extractBoard():
