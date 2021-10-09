# Authors: Keanan, Zach, Vince
# License Plate Detection

import cv2 as cv
import numpy as np
import imutils

# access camera through gstreamer pipeline
capture = cv.VideoCapture(
    "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"
)


# This loop saves a single image for use in original algorithm 
"""
ret,frame = cap.read()
while(True):
    cv2.imshow('img1',frame) #display the captured image
    if cv2.waitKey(1) & 0xFF == ord('y'): #save on pressing 'y' 
        cv2.imwrite('images/c1.png',frame)
        cv2.destroyAllWindows()
        break
"""

while True:
    ret, frame = capture.read()

    # Video feed transformations ------------------------------------
    # convert color to gray
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # noise reduction using bilateral filter, like a gaussian blur
    bfilter = cv.bilateralFilter(gray, 11, 17, 17)

    # use Canny algorithm to find edges
    edged = cv.Canny(bfilter, 30, 200)

    # LP algorithm --------------------------------------------------
    # grab rectangular points ie. license plate
    # need to figure out whats going on here
    rectPoints = cv.findContours(edged.copy(), cv.RETR_TREE,
                                 cv.CHAIN_APPROX_SIMPLE)
    contour = imutils.grab_contours(rectPoints)
    contour = sorted(contour, key=cv.contourArea, reverse=True)[:10]

    # determine if we have the rectangle
    location = None
    for contour in contour:
        # save list (array) of contours
        approx = cv.approxPolyDP(contour, 10, True)
        # if its
        if len(approx) == 4:
            location = approx
            break

    """
    # here is where the code will break without tracking
    mask = np.zeros(gray.shape, np.uint8)
    new_frame = cv.drawContours(mask, [location], 0, 255, -1)
    new_frame = cv.bitwise_and(new_frame, new_frame, mask=mask)
    """
    
    #print(location)

    # attempt at tracking rectangle around lp,
    # this will display a rectangle around recPoints (ideally)
    #cv.rectangle(frame, location[0], location[3], (0,255,0), -1)

    # display video feeds--------------------------------------------
    #cv.imshow('frame', frame)
    cv.imshow('edged', edged)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera
capture.release()
# Close windows
cv.destroyAllWindows