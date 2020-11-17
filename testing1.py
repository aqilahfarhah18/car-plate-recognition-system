import cv2
import pytesseract
import numpy as np
from skimage.segmentation import clear_border
import imutils
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

image = cv2.imread('C:\\opencv-anpr\\license_plates\\group1\\5.jpg')
image=cv2.resize(image,(620,480))
cv2.imshow('OriImage',image)
cv2.waitKey(0)
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',gray)
cv2.waitKey(0)
gray = cv2.bilateralFilter(gray, 13, 25, 25)
cv2.imshow('blur',gray)
cv2.waitKey(0)

rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKern)
cv2.imshow("tophat", tophat)
cv2.waitKey(0)

squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
light = cv2.threshold(light, 0, 255, cv2.THRESH_BINARY| cv2.THRESH_OTSU)[1]
cv2.imshow("light regions", light)
cv2.waitKey(0)

gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F,
			dx=1, dy=0, ksize=0)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
gradX = gradX.astype("uint8")
cv2.imshow("gradX", gradX)
cv2.waitKey(0)

gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
thresh = cv2.threshold(gradX, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow('thresh1',thresh)
cv2.waitKey(0)

thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=1)
cv2.imshow('thresh2',thresh)
cv2.waitKey(0)

thresh = cv2.bitwise_and(thresh, thresh, mask=light)
thresh = cv2.dilate(thresh, None, iterations=2)
thresh = cv2.erode(thresh, None, iterations=1)
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)

edged = cv2.Canny(gray, 30, 200) #Perform Edge detection
cv2.imshow('edge',edged)
cv2.waitKey(0)

# contours=cv2.findContours(edged.copy(),cv2.RETR_TREE,
#                                             cv2.CHAIN_APPROX_SIMPLE)
# contours = imutils.grab_contours(contours)
# contours = sorted(contours,key=cv2.contourArea, reverse = True)[:20]
#
# screenCnt = None
#
#
# for c in contours:
#     # approximate the contour
#     peri = cv2.arcLength(c, True)
#     approx = cv2.approxPolyDP(c, 0.018 * peri, True)
#     # if our approximated contour has four points, then
#     # we can assume that we have found our screen
#     if len(approx) == 4:
#         screenCnt = approx
#         break
# if screenCnt is None:
#     detected = 0
#     print ("No contour detected")
# else:
#      detected = 1
#
# if detected == 1:
#     cv2.drawContours(image, [screenCnt], -1, (0, 0, 255), 3)
#




#
# gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
# gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
# thresh = cv2.threshold(gradX, 0, 255,
#     cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# cv2.imshow("gradX1", gradX)
# cv2.waitKey(0)
#
# thresh = cv2.erode(thresh, None, iterations=2)
# thresh = cv2.dilate(thresh, None, iterations=2)
# cv2.imshow("thresh", thresh)
# cv2.waitKey(0)
#
# thresh = cv2.bitwise_and(thresh, thresh, mask=light)
# thresh = cv2.dilate(thresh, None, iterations=2)
# thresh = cv2.erode(thresh, None, iterations=1)
# cv2.imshow("thresh1", thresh)
# cv2.waitKey(0)
