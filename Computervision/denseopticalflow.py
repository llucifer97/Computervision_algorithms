#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 18:33:50 2018

@author: ayush
"""
import cv2

import numpy as np

cap = cv2.VideoCapture('slow.flv')


ret,first_frame = cap.read()

previous_gray = cv2.cvtColor(first_frame,cv2.COLOR_BGR2GRAY)

hsv = np.zeros_like(first_frame)

hsv[...,1] = 255

while True:
    
    ret,frame2 = cap.read()
    
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    
    flow = cv2.calcOpticalFlowFarneback(previous_gray,next,None,1,3,15,5,1,0)
    
    magnitude,angle = cv2.cartToPolar(flow[...,0],flow[...,1])
    
    hsv[...,0] = angle * (180/(np.pi/2))
    hsv[...,2] = cv2.normalize(magnitude,None,0,255,cv2.NORM_MINMAX)
    
    final = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    
    
    
    cv2.imshow('dense optical flow',final)
    
    if cv2.waitKey(1) == 13:
        break
    
    previous_gray = next
    
cap.release()

cv2.destroyAllWindows()
    
    














































