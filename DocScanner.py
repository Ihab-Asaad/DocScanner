import cv2
import numpy as np

widthImg = 640
heighImg = 480

frameWidth = 640
frameHeight = 480

path = r'H:\AIS\Computer Vision\DocScanner\scanImg.jpg'

## to capture from camera
# cap = cv2.VideoCapture(0)
# cap.set(3, widthImg)
# cap.set(4,heighImg)
# cap.set(10,150) # Brightness of the image


def getContours(img):
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
            #cv2.drawContours(imgContour, cnt, -1 ,(255,0,0),3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
            #x,y,w,h = cv2.boundingRect(approx)
    # cv2.drawContours(imgContour, biggest, -1 ,(255,0,0),20)
    return biggest


def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)

    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[2] = myPoints[np.argmax(add)]

    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[3] = myPoints[np.argmax(diff)]
    return myPointsNew


def preProcessing(img):
    """
    Preprocess the image to detect edges.
    """
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 150, 200)
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=2)
    imgThres = cv2.erode(imgDial, kernel, iterations=1)
    return imgThres


def rescale(biggest, ratioWidth, rationHeight):
    biggest[:, 0, 0] = biggest[:, 0, 0] * (ratioWidth)
    biggest[:, 0, 1] = biggest[:, 0, 1] * (rationHeight)
    return biggest


def getWarp(orgImg, biggest, orgHeight, orgWidth, ratioWidth, ratioHeight):
    """
    Apply a perspective transformation to an image.
    """
    biggest = reorder(biggest)
    biggest = rescale(biggest, ratioWidth, ratioHeight)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [orgWidth, 0], [orgWidth, orgHeight],
                       [0, orgHeight]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    imgOutput = cv2.warpPerspective(orgImg, matrix, (orgWidth, orgHeight))

    imgCropped = imgOutput[30:imgOutput.shape[0] - 30,
                           30:imgOutput.shape[1] - 30]
    imgCropped = cv2.resize(imgCropped, (orgWidth, orgHeight))
    return imgCropped


def crop_doc(path):
    """
    Read the image
    Resize the image to (widthImg = 640, heighImg = 480) to reduce the number of operations while finding the document in the image
    Preprocess the image to detect edges
    Find Contours and select the biggest one, which must be the biggest Contour
    Apply a perspective transformation to an image.
    """
    img = cv2.imread(path)
    orgHeight, orgWidth, orgChannel = img.shape
    ratioWidth, ratioHeight = orgWidth / widthImg, orgHeight / heighImg
    imgCopy = img.copy()
    img = cv2.resize(img, (widthImg, heighImg))
    # img = cv2.resize(img, (1500,1000))
    # imgContour = img.copy()

    imgThres = preProcessing(img)
    biggest = getContours(imgThres)
    imgWarped = getWarp(imgCopy, biggest, orgHeight, orgWidth, ratioWidth,
                        ratioHeight)
    return imgWarped


## to run this file independtly, uncomment the following lines:
# while True:
#     # success, img = cap.read() # for catpuring from camera
#     img = cv2.imread(path)
#     orgHeight, orgWidth , orgChannel = img.shape
#     ratioWidth, rationHeight = orgWidth/widthImg , orgHeight/heighImg
#     imgCopy = img.copy()
#     img = cv2.resize(img, (widthImg,heighImg))
#     imgContour = img.copy()

#     imgThres = preProcessing(img)
#     biggest = getContours(imgThres)
#     imgWarped = getWarp(imgCopy, biggest)
#     cv2.imshow("Result",cv2.resize(imgWarped,(1200,1000)))
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break