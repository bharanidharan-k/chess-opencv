import cv2
import numpy as np
from matplotlib import pyplot as plt



def warp_image():
    img = cv2.imread('/home/kb/PycharmProjects/ocv/cb4.jpg')
    gray = cv2.imread('/home/kb/PycharmProjects/ocv/cb4.jpg', cv2.COLOR_BGR2GRAY)
    # gray = cv2.resize(gray, (300,300))
    gray = cv2.medianBlur(gray,5)
    # ratio = gray.shape[0] / 300.0

    # cv2.imshow('test', inp)
    #ret, thresh = cv2.threshold(gray, 127, 255, 0)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                cv2.THRESH_BINARY, 11, 2)
    _, contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for cnt in contours:
        if cv2.contourArea(cnt) > 15000 and cv2.contourArea(cnt) < 130000:  # remove small areas like noise etc
            print(cv2.contourArea(cnt))
            hull = cv2.convexHull(cnt)  # find the convex hull of contour
            hull = cv2.approxPolyDP(hull, 0.1 * cv2.arcLength(hull, True), True)
            # [[[590 456]] [[57 460]]  [[162 197]] [[477 179]]]
            if len(hull) == 4:
                cv2.drawContours(gray, [hull], 0, (0, 0, 0), 3)
                break

    # cv2.imshow('img', gray)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # now that we have our screen contour, we need to determine
    # the top-left, top-right, bottom-right, and bottom-left
    # points so that we can later warp the image -- we'll start
    # by reshaping our contour to be our finals and initializing
    # our output rectangle in top-left, top-right, bottom-right,
    # and bottom-left order
    pts = hull.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point has the smallest sum whereas the
    # bottom-right has the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # compute the difference between the points -- the top-right
    # will have the minumum difference and the bottom-left will
    # have the maximum difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # multiply the rectangle by the original ratio
    # rect *= ratio

    # now that we have our rectangle of points, let's compute
    # the width of our new image
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    # ...and now for the height of our new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    # take the maximum of the width and height values to reach
    # our final dimensions
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))

    # construct our destination points which will be used to
    # map the screen to a top-down, "birds eye" view
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # calculate the perspective transform matrix and warp
    # the perspective to grab the screen
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return cv2.resize(warp,(400,400))



def separateSquares():
    gBoard = warp_image()
    # gBoard = cv2.medianBlur(gBoard,3)
    # kernel_sharpen_1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # gBoard = cv2.filter2D(gBoard, -1, kernel_sharpen_1)
    thresh = cv2.adaptiveThreshold(gBoard, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                   cv2.THRESH_BINARY, 7, 3)

    cv2.imshow('thresh', thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours:
        if cv2.contourArea(cnt) > 0 and cv2.contourArea(cnt) < 500:  # remove small areas like noise etc
            print(cv2.contourArea(cnt))
            hull = cv2.convexHull(cnt)  # find the convex hull of contour
            hull = cv2.approxPolyDP(hull, 0.1 * cv2.arcLength(hull, True), True)
            # [[[590 456]] [[57 460]]  [[162 197]] [[477 179]]]
            if len(hull) <= 4:
                cv2.drawContours(gBoard, [hull], 0, (0, 0, 0), 2)
                # break

    cv2.imshow('final', gBoard)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def identifySmallSquares_Contours():
    board = warp_image()
    # blurred = cv2.pyrMeanShiftFiltering(board.copy(),19,19)
    gBoard = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    # gBoard = cv2.medianBlur(gBoard, 3,3)
    thresh = cv2.adaptiveThreshold(gBoard, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                   cv2.THRESH_BINARY, 5, 3)
    _,contours,_ = cv2.findContours(thresh,cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours:
        if cv2.contourArea(cnt) > 0 and cv2.contourArea(cnt) < 200:  # remove small areas like noise etc
            print(cv2.contourArea(cnt))
            hull = cv2.convexHull(cnt)  # find the convex hull of contour
            hull = cv2.approxPolyDP(hull, 0.2 * cv2.arcLength(hull, True), True)
            # [[[590 456]] [[57 460]]  [[162 197]] [[477 179]]]
            if len(hull) <= 4:
                cv2.drawContours(board, [hull], 0, (0, 0, 200), 2)
                pts = hull.reshape(4, 2)
                rect = np.zeros((4, 2), dtype="float32")

                # the top-left point has the smallest sum whereas the
                # bottom-right has the largest sum
                s = pts.sum(axis=1)
                rect[0] = pts[np.argmin(s)]
                rect[2] = pts[np.argmax(s)]

                # compute the difference between the points -- the top-right
                # will have the minumum difference and the bottom-left will
                # have the maximum difference
                diff = np.diff(pts, axis=1)
                rect[1] = pts[np.argmin(diff)]
                rect[3] = pts[np.argmax(diff)]

                # multiply the rectangle by the original ratio
                # rect *= ratio

                # now that we have our rectangle of points, let's compute
                # the width of our new image
                (tl, tr, br, bl) = rect
                widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
                widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

                # ...and now for the height of our new image
                heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
                heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

                # take the maximum of the width and height values to reach
                # our final dimensions
                maxWidth = max(int(widthA), int(widthB))
                maxHeight = max(int(heightA), int(heightB))

                # construct our destination points which will be used to
                # map the screen to a top-down, "birds eye" view
                dst = np.array([
                    [0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]], dtype="float32")

                # calculate the perspective transform matrix and warp
                # the perspective to grab the screen
                M = cv2.getPerspectiveTransform(rect, dst)
                warp = cv2.warpPerspective(board, M, (maxWidth, maxHeight))
                cv2.imshow('small', cv2.resize(warp, (50, 50)))

    cv2.imshow('final', board)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def identifySmallSquares_Hough():
    board = warp_image()
    gBoard = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gBoard, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                   cv2.THRESH_BINARY, 5, 3)

    edges = cv2.Canny(thresh, 50, 240, apertureSize=7)
    cv2.imshow('thresh.jpg', thresh)
    cv2.imshow('edges.jpg', edges)

    lines = cv2.HoughLines(edges, 0.2, np.pi / 180, 240)
    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(board, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #
    # minLineLength = 10
    # maxLineGap = 50
    # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)
    # print(edges.shape)
    print(lines[0])
    # for x1, y1, x2, y2 in lines[0]:
    #     cv2.line(board, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('houghlines3.jpg', board)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # plt.subplot(121), plt.imshow(thresh, cmap='gray')
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122), plt.imshow(edges, cmap='gray')
    # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    #
    #
    # plt.show()

def drawGrid():
    gBoard = warp_image()
    for i in range(8):
        cv2.line(gBoard, (0, i*50), (400, i*50),(0,0,0,), 1,1)
        cv2.line(gBoard, (i*50, 0), (i*50,400),(0,0,0,), 1,1)


    cv2.imshow('final', gBoard)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


separateSquares()