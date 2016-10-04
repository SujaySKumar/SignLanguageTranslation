import cv2
import numpy as np
from matplotlib import pyplot as plt


def create_feature_vector(contours):
    vec = []
    for contour in contours:
        if contour is not None:
            flattened_contour = contour.flatten()
            flattened_contour = flattened_contour.astype('float64')
            flattened_contour = flattened_contour/flattened_contour.max()
            vec = vec+flattened_contour.tolist()
    return vec


final_feature_vector = []
cap = cv2.VideoCapture('test2.mov')
while(cap.isOpened()):
    #Read the frame
    ret, frame = cap.read()

    if ret == False:
        break

    #Convert the image to gray
    img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Crop the image in 2 vertical halves for videos with front and side view in the same video.
    y_coords = img1.shape[0]
    x_coords = img1.shape[1]

    #Leave out 20 pixels in the top to exclude the numbers
    img = img1[20:y_coords/2, 0:x_coords]

    # global thresholding
    #ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

    # Otsu's thresholding
    #ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img,(5,5),0)
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    #To find out the contour with the maximum area. The border comes out to be the contour with
    #the largest area. Hence remove the largest and get all other contours.
    max_area = 0
    _, contours, hierarchy = cv2.findContours(th3,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    hulls = []
    areas_to_exclude = []
    for i in range(len(contours)):
        cnt=contours[i]
        area = cv2.contourArea(cnt)

        #Excluding areas which are too small
        if area<30:
            areas_to_exclude.append(i)
        hulls.append(cv2.convexHull(cnt))
        if(area>max_area):
            max_area=area
            ci=i

    #Contour with highest area
    cnt=contours[ci]
    areas_to_exclude.append(ci)

    for indices in areas_to_exclude:
        contours[indices]=None

    #Convex hull of the highest area contour
    #hull = cv2.convexHull(cnt)
    drawing = np.zeros(img.shape)

    #Draw all contours
    cv2.drawContours(frame,contours,-1,(0,255,0),2)

    #Draw all convex hulls
    #cv2.drawContours(frame,hulls,-1,(0,0,255),2)


    cv2.imshow('output',frame)
    cv2.imshow('input', img)

    frame_feature_vector = create_feature_vector(contours)
    print len(frame_feature_vector)
    final_feature_vector.append(frame_feature_vector)
    print "final-----------------", len(final_feature_vector)

    cv2.waitKey()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
print "whaaat-------------", len(final_feature_vector)
