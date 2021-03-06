#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 08:23:49 2018

@author: ayush
"""
import numpy as np
import cv2

image = cv2.imread('vishal.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

cv2.imshow('original image',image)
cv2.waitKey(0)

ret,thresh = cv2.threshold(gray,176,255,0)

_,contours,hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

n = len(contours)  - 1 
contours = sorted(contours, key = cv2.contourArea,reverse = False)[:n]

for c in contours:
    hull = cv2.convexHull(c)
    cv2.drawContours(image,[hull],0,(0,255,0),2)
    cv2.imshow('convex hull',image)

cv2.waitKey(0)
cv2.destroyAllWindows()






































































