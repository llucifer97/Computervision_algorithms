#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 09:27:05 2018

@author: ayush
"""

import cv2
import numpy as np

template = cv2.imread('dog.jpg',0)
cv2.imshow('template',template)
cv2.waitKey()

target = cv2.imread('vishal.jpg')
target_gray = cv2.cvtColor(target,cv2.COLOR_BGR2GRAY)



ret,thresh1 = cv2.threshold(template,127,255,0)
ret,thresh2 = cv2.threshold(target_gray,127,255,0)

_,contours,hierarchy = cv2.findContours(thresh1.copy(),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

sorted_contours = sorted(contours,key = cv2.contourArea, reverse = True)

template_contour = contours[1]

_,contours,hierarchy = cv2.findContours(thresh2.copy(),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    match = cv2.matchShapes(template_contour,c,1,0.0)
    print('match')
    
    if match < 0.15:
        closest_contour = c
    else:
        closest_contour = []
        
cv2.drawContours(target,[closest_contour],-1,(0,255,0),3)##
cv2.imshow('output',target)
cv2.waitKey(0)
cv2.destroyAllWindows()









































































