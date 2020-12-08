import pytesseract
from PIL import Image
import cv2
import numpy as np
import csv
from pytesseract import Output

#resize image
#def resize(image):
#    return cv2.resize(image, (1350, 1150))

#skew correction
def deskew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)    
    return rotated

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
# noise removal
#def remove_noise(image):
#    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    # threshold the image, setting all foreground pixels to
    # 255 and all background pixels to 0
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    #return cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            #cv2.THRESH_BINARY,11,2)
    #return cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            #cv2.THRESH_BINARY,11,2)


def deskew_gray(image):
    img = deskew(image)
    img_1 = get_grayscale(img)
    return img_1
        
def deskew_gray_thresh(image):
    img = deskew(image)
    img_1 = get_grayscale(img)
    img_2 = thresholding(img_1)
    return img_2


#erosion
#def erode(image):
#    kernel = np.ones((5,5),np.uint8)
#    return cv2.erode(image, kernel, iterations = 1)

#dilation
#def dilate(image):
#    kernel = np.ones((5,5),np.uint8)
#    return cv2.dilate(image, kernel, iterations = 1)
    

##opening - erosion followed by dilation
#def opening(image):
#    kernel = np.ones((5,5),np.uint8)
#    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

##canny edge detection
#def canny(image):
#    return cv2.Canny(image, 100, 200)

##template matching
#def match_template(image, template):
#    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED) 