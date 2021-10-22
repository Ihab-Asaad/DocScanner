import cv2
from matplotlib.cbook import maxdict
import numpy as np

# widthImg = 640
# heightImg = 480

widthImg = 640
heightImg = 640

frameWidth = 640
frameHeight = 480

path = r'H:\AIS\Computer Vision\DocScanner\scanImg.jpg'

## to capture from camera
# cap = cv2.VideoCapture(0)
# cap.set(3, widthImg)
# cap.set(4,heightImg)
# cap.set(10,150) # Brightness of the image


def simplify_contour(contour, n_corners=4):
    '''
    Binary searches best `epsilon` value to force contour 
        approximation contain exactly `n_corners` points.

    :param contour: OpenCV2 contour.
    :param n_corners: Number of corners (points) the contour must contain.

    :returns: Simplified contour in successful case. Otherwise returns initial contour.
    '''
    n_iter, max_iter = 0, 50
    lb, ub = 0., 1.

    while True:
        n_iter += 1
        if n_iter > max_iter:
            return contour

        k = (lb + ub) / 2.
        eps = k * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, eps, True)

        if len(approx) > n_corners:
            lb = (lb + ub) / 2.
        elif len(approx) < n_corners:
            ub = (lb + ub) / 2.
        else:
            return approx, eps


def getContours(imgContour, img):
    """
    Find Contours and select the biggest one, which must be the biggest Contour.
    """
    biggest = np.array([])
    maxArea = 0

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            approx, eps = simplify_contour(cnt, n_corners=4)
            # peri = cv2.arcLength(cnt, True)
            # approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
            x, y, w, h = cv2.boundingRect(approx)
    cv2.drawContours(imgContour, biggest, -1, (255, 0, 0), 20)
    cv2.imwrite("imgContour.jpg", cv2.resize(imgContour, None, fx=1, fy=1))
    return biggest


def reorder_min_max(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)

    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[2] = myPoints[np.argmax(add)]

    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[3] = myPoints[np.argmax(diff)]

    return myPointsNew


def calc_dist(points, idx):
    """
    Return sorted euclidean distance between point with given idx and other points
    """
    lst = [(points[i][0] * points[idx][0] + points[i][1] * points[idx][1], i)
           for i in range(4) if i != idx]
    lst_sorted = sorted(lst)
    return lst_sorted


def reorder_min_max_euclidean(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    minheight_idx = np.argmin(myPoints[:, 1])
    sorted_lst = calc_dist(myPoints, minheight_idx)
    if (myPoints[sorted_lst[0][1], 0] > myPoints[minheight_idx, 0]):
        myPointsNew[0] = myPoints[minheight_idx]
        myPointsNew[1] = myPoints[sorted_lst[0][1]]
        myPointsNew[2] = myPoints[sorted_lst[2][1]]
        myPointsNew[3] = myPoints[sorted_lst[1][1]]
    else:
        myPointsNew[1] = myPoints[minheight_idx]
        myPointsNew[0] = myPoints[sorted_lst[0][1]]
        myPointsNew[2] = myPoints[sorted_lst[2][1]]
        myPointsNew[3] = myPoints[sorted_lst[1][1]]
    return myPointsNew, int(np.sqrt(sorted_lst[0][0])), int(
        np.sqrt(sorted_lst[1][0]))


def reorder(myPoints):
    # myPointsNew = reorder_min_max(myPoints)
    myPointsNew, width, height = reorder_min_max_euclidean(myPoints)
    return myPointsNew, width, height


def preProcessing(img):
    """
    Preprocess the image to detect edges.
    """
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)

    imgCanny = cv2.Canny(imgBlur, 100, 175)

    kernel = np.ones((3, 3))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=2)
    imgThres = cv2.erode(imgDial, kernel, iterations=2)
    return imgThres


def rescale(biggest, ratioWidth, rationHeight):
    biggest[:, 0, 0] = biggest[:, 0, 0] * (ratioWidth)
    biggest[:, 0, 1] = biggest[:, 0, 1] * (rationHeight)
    return np.int32(biggest)


def getWarp(orgImg, biggest, ratioWidth, ratioHeight):
    """
    Apply a perspective transformation to an image.
    """
    biggest = rescale(biggest, ratioWidth, ratioHeight)
    biggest, width, height = reorder(biggest)
    test_img = orgImg.copy()
    cv2.drawContours(test_img, biggest, -1, (255, 0, 0), 20)
    pts1 = np.float32(biggest)
    # newWidth = orgWidth
    newWidth = min(int(width) * 2, 3 * widthImg)
    # newHeight = orgHeight
    newHeight = min(int(height) * 2, 3 * heightImg)
    pts2 = np.float32([[0, 0], [newWidth, 0], [newWidth, newHeight],
                       [0, newHeight]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    # # Use homography
    # h, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransacReprojThreshold=1.0)
    # imgOutput = cv2.warpPerspective(orgImg, h, (newWidth, newHeight))

    imgOutput = cv2.warpPerspective(orgImg,
                                    matrix, (newWidth, newHeight),
                                    flags=cv2.INTER_LINEAR)
    imgCropped = imgOutput[20:imgOutput.shape[0] - 20,
                           20:imgOutput.shape[1] - 20]
    # imgCropped = cv2.resize(imgCropped, (newWidth, orgHeight))
    return imgCropped


def crop_doc(path):
    """
    Read the image
    Resize the image to (widthImg = 640, heightImg = 480) to reduce the number of operations while finding the document in the image
    Preprocess the image to detect edges
    Find Contours and select the biggest one, which must be the biggest Contour
    Apply a perspective transformation to an image.
    """
    img = cv2.imread(path)
    orgHeight, orgWidth, _ = img.shape
    ratioWidth, ratioHeight = orgWidth / widthImg, orgHeight / heightImg
    imgCopy = img.copy()
    img = cv2.resize(img, (widthImg, heightImg))
    imgContour = img.copy()

    imgThres = preProcessing(img)
    biggest = getContours(imgContour, imgThres)
    imgWarped = getWarp(imgCopy, biggest, ratioWidth, ratioHeight)
    return imgWarped
