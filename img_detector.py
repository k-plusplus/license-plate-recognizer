# Authors: Keanan
# License Plate Detection

from logging import NullHandler
import cv2 as cv
import numpy as np
import imutils
#import easyocr

# access camera through gstreamer pipeline

capture = cv.VideoCapture(
    "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"
) 


directory = r'/home/zachary/Desktop/detector'

# This loop saves a single image for use in original algorithm 


""" while True:
    ret,frame = capture.read()
    cv.imshow('img1',frame) #display the captured image
    if cv.waitKey(1) & 0xFF == ord('y'): #save on pressing 'y' 
        cv.imwrite(directory+'/img1.jpg',frame)
        print("Image save successful\n")
        cv.destroyAllWindows()
        break """


while True:
    ret, frame = capture.read()
    cv.imwrite(directory+'/img1.jpg',frame)
    img = cv.imread('/home/zachary/Desktop/detector/img1.jpg')

    
    # Video feed transformations ------------------------------------
    # convert color to gray
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # noise reduction using bilateral filter, like a gaussian blur
    bfilter = cv.bilateralFilter(gray, 11, 17, 17)

    # use Canny algorithm to find edges
    edged = cv.Canny(bfilter, 0, 545)

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

    if location is not None:
        mask = np.zeros(gray.shape, np.uint8)
        new_frame = cv.drawContours(mask, [location], 0, 255, -1)
        new_frame = cv.bitwise_and(new_frame, new_frame, mask=mask)
        (x, y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped = gray[x1:x2+1, y1:y2+1]
        cv.imshow('cropped', cv.cvtColor(cropped, cv.COLOR_BGR2RGB))
        #print(location)
    #print(location)
    # here is where the code will break without tracking
        
    # display video feeds--------------------------------------------
    #cv.imshow('frame', frame)
    #cv.imshow('edged', edged)

    #cv.drawContours(img, location, -1, (0,255,0), 3)
    #cv.imshow('edged', edged)
        #cv.imshow('new_frame', new_frame)
        #cv.imshow('original', img)
	#cv.imshow('cropped',cv.cvtColor(cropped, cv.COLOR_BGR2RGB))
    # attempt at tracking rectangle around lp,
    # this will display a rectangle around recPoints (ideally)
    #rec_mask = cv.rectangle(img, (,location[0][1]), (location[3][0],location[3][1]), (0,255,0), -1)
    #cv.imshow('rectangle mask', rec_mask)


    

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera
capture.release()
# Close windows
cv.destroyAllWindows
