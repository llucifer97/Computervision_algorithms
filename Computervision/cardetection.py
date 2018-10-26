#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 12:23:16 2018

@author: ayush
"""
import cv2
import time 
import numpy as np


car_classifier = cv2.CascadeClassifier('car haar cascade.xml')


cap = cv2.VideoCapture('slow.flv')


while cap.isOpened():
    time.sleep(.05)
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    
    cars = car_classifier.detectMultiScale(gray,1.3,3)

    for (x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        cv2.imshow('cars',frame)

    if cv2.waitKey(1) == 13:
        break
cap.release()
cv2.destroyAllWindows()



















































