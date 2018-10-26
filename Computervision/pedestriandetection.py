#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 11:57:42 2018

@author: ayush
"""

import cv2
import numpy as np

body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')

cap = cv2.VideoCapture('pedestrian.mp4')


while cap.isOpened():
    
    ret,frame = cap.read()
    frame = cv2.resize(frame,None,fx = 0.5,fy = 0.5,interpolation = cv2.INTER_LINEAR)
    
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    bodies = body_classifier.detectMultiScale(gray,1.2,3)
    
    for (x,y,w,h) in bodies:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        cv2.imshow('pedestrians',frame)

    if cv2.waitKey(1) == 13:
        break
    
cap.release()
cv2.destroyAllWindows()
        















































































