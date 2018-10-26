#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 14:48:09 2018

@author: ayush
"""
import cv2
import numpy as np

cap = cv2.VideoCapture(0)


lower_purple = np.array([125,0,0])
upper_purple = np.array([175,255,255])

while True:
    
    ret,frame = cap.read()
    
    hsv_img = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(hsv_img,lower_purple,upper_purple)

    res = cv2.bitwise_and(frame,frame,mask = mask)
    
    cv2.imshow('Original',frame)

    cv2.imshow('filtered color only',res)
    
    if cv2.waitKey(1) == 13:
        break

cap.release()















































