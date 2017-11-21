from typing import re

import cv2
import numpy as np



def warp_image():
    gray = cv2.imread('/home/kb/PycharmProjects/ocv/cb4.jpg', cv2.COLOR_BGR2GRAY)
    # cv2.imshow('test', inp)
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    _, contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        print(cv2.contourArea(cnt) )
        if cv2.contourArea(cnt) > 1000:  # remove small areas like noise etc
            hull = cv2.convexHull(cnt)  # find the convex hull of contour
            hull = cv2.approxPolyDP(hull, 0.1 * cv2.arcLength(hull, True), True)
            if len(hull) == 4:
                cv2.drawContours(gray, [hull], 0, (0, 100, 100), 3)

    cv2.imshow('img', gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_corner():
    img = cv2.imread('/home/kb/PycharmProjects/ocv/cb4.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)

    dst = cv2.cornerHarris(gray, 4, 5, 0.04)  # to detect only sharp corners
    # dst = cv2.cornerHarris(gray, 14, 5, 0.04)    # to detect soft corners

    # Result is dilated for marking the corners
    dst = cv2.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst > 0.02 * dst.max()] = [0, 0, 255]

    cv2.imshow('Harris Corners', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def extract_roi():
    img = cv2.imread('/home/kb/PycharmProjects/ocv/cb4.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)

    dst = cv2.cornerHarris(gray, 4, 5, 0.04)  # to detect only sharp corners
    # dst = cv2.cornerHarris(gray, 14, 5, 0.04)    # to detect soft corners

    # Result is dilated for marking the corners
    dst = cv2.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst > 0.02 * dst.max()] = [0, 0, 255]

    cv2.imshow('Harris Corners', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

detect_corner()



