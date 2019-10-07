#!/usr/bin/env python

import numpy as np
import cv2

# Set Capture Device as Webcam
cap = cv2.VideoCapture(0)
"""
        def greenCircleDetect(frame)
            Returns frame with contour drawn around all Green Objects.
            Requires fr(frame) to work on.
"""
def greenCircleDetect(fr):
    # UpperBound and LowerBound HSV Range for GREEN colour.
    lowerBound = np.array([33, 80, 40])
    upperBound = np.array([102, 255, 255])
    # First Convert Frame to HSV
    imgHSV = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    # Now Create a Mask for GREEN Colour
    mask = cv2.inRange(imgHSV, lowerBound, upperBound)
    # Now BitwiseAND HSV and mask
    finalFrame = cv2.bitwise_and(imgHSV, imgHSV, mask=mask);
    # Convert final frame to grayscale before contour detection.
    gray_image = cv2.cvtColor(finalFrame, cv2.COLOR_BGR2GRAY)
    # create a thresh image out of grayscale image provided.
    ret, thresh = cv2.threshold(gray_image, 127, 255, 0)
    # detect contours and draw border around it with black(0,0,0) colour.
    ctrs, hrc = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(fr, ctrs, -1, (0, 0, 0), 2)
    return fr

"""
        def redCircleDetect(frame)
            Returns frame with contour drawn around all Red Objects.
            Requires fr(frame) to work on.
"""
def redCircleDetect(fr):
    # UpperBound and LowerBound HSV Range for Red colour for Lower Saturation Range.
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    # First Convert Frame to HSV
    imgHSV = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    # Now Create a Mask for Low Saturation Blue colour.
    mask0 = cv2.inRange(imgHSV, lower_red, upper_red)
    # UpperBound and LowerBound HSV Range for Red colour for Higher Saturation Range.
    lower_red = np.array([170, 50, 50])
    upper_red = np.array([180, 255, 255])
    # Now Create a Mask for High Saturation Blue colour.
    mask1 = cv2.inRange(imgHSV, lower_red, upper_red)
    #add Both the masks to make complete mask covering both high and low saturation ranges
    mask = mask1 + mask0;
    # Mask it Out
    finalFrame = cv2.bitwise_and(imgHSV, imgHSV, mask=mask);
    # GreyScale Conversion
    gray_image = cv2.cvtColor(finalFrame, cv2.COLOR_BGR2GRAY)
    # threshold conversion
    ret, thresh = cv2.threshold(gray_image, 127, 255, 0)
    # contour detection
    ctrs, hrc = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(fr, ctrs, -1, (0, 0, 0), 2)
    return fr

"""
        def blueCircleDetect(frame)
            Returns frame with contour drawn around all Blue Objects.
            Requires fr(frame) to work on.
"""

def blueCircleDetect(fr):
    # UpperBound and LowerBound HSV Range for Blue colour.
    lowerBound = np.array([100, 150, 0])
    upperBound = np.array([140, 255, 255])
    # First Convert Frame to HSV
    imgHSV = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    # Now Create a Mask for GREEN Colour
    mask = cv2.inRange(imgHSV, lowerBound, upperBound)
    # Now BitwiseAND HSV and mask
    finalFrame = cv2.bitwise_and(imgHSV, imgHSV, mask=mask);
    # Convert final frame to grayscale before contour detection.
    gray_image = cv2.cvtColor(finalFrame, cv2.COLOR_BGR2GRAY)
    # create a thresh image out of grayscale image provided.
    ret, thresh = cv2.threshold(gray_image, 127, 255, 0)
    # detect contours and draw border around it with black(0,0,0) colour.
    ctrs, hrc = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(fr, ctrs, -1, (0, 0, 0), 2)
    return fr


while True:
    # Get a frame from Webcam.
    ret, frame = cap.read()
    # Process the frame for Green-->Red-->Blue colours and draw border around them.
    frame=blueCircleDetect(redCircleDetect(greenCircleDetect(frame)))
    #Display the final frame
    cv2.imshow('frame', frame)
    cv2.waitKey(3)

cap.release()
cv2.destroyAllWindows()
