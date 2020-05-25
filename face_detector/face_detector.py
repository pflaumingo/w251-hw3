import numpy as np
import cv2 as cv
import sys
print(sys.path)

# 1 should correspond to /dev/video1 , your USB camera. The 0 is reserved for the TX2 onboard camera
cap = cv.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # We don't use the color information, so might as well save space
    face_cascade = cv.CascadeClassifier('/Users/npflaum/anaconda3/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        face = gray[y:y+h, x:x+w]
        cv.imshow('frame', face)
        # rc, png = cv.imencode('.png', face)

    # cv.imshow('frame', gray)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    # print(gray)
    # face detection and other logic goes here