import cv2
import numpy as np


def resize_image(image, basewidth):
    wpercent = (basewidth / float(image.shape[1]))
    hsize = int(image.shape[0] * float(wpercent))
    return cv2.resize(image, (basewidth, hsize), interpolation=cv2.INTER_CUBIC)


def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 3)


#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


#dilation
def dilate(image):
    kernel = np.ones((2, 2), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


#erosion
def erode(image):
    kernel = np.ones((2, 2), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)


#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image,
                             M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated


#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)


# Equalizaiton histogram:
def equalize_hist(gray):
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4, 4))
    equalized = clahe.apply(gray)
    # equalized = cv2.equalizeHist(gray)
    return equalized


def sharping(image):
    gaussian_filter = cv2.GaussianBlur(image, (3, 3), 0.5, 0.5)
    sharp_image = cv2.addWeighted(image, 1.5, gaussian_filter, -0.5, 0)
    return sharp_image