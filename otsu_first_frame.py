import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture('test.mov')
while(cap.isOpened()):
    #Read the frame
    ret, frame = cap.read()

    #Convert the image to gray
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # global thresholding
    #ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

    # Otsu's thresholding
    #ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img,(5,5),0)
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    #To find out the contour with the maximum area. Here this is irrelevant as the haed comes out with
    #the largest area. Hence get all contours
    max_area = 0
    _, contours, hierarchy = cv2.findContours(th3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    hulls = []
    # for i in range(len(contours)):
    #     cnt=contours[i]
    #     area = cv2.contourArea(cnt)
    #     hulls.append(cv2.convexHull(cnt))
    #     if(area>max_area):
    #         max_area=area
    #         ci=i

    #Contour with highest area
    #cnt=contours[ci]

    #Convex hull of the highest area contour
    #hull = cv2.convexHull(cnt)
    drawing = np.zeros(img.shape)

    #Draw all contours
    cv2.drawContours(frame,contours,-1,(0,255,0),2)

    #Draw all convex hulls
    cv2.drawContours(frame,hulls,-1,(0,0,255),2)


    cv2.imshow('output',frame)
    cv2.imshow('input', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
