#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 08:02:23 2018

@author: ayush
"""
import numpy as np
import cv2

image = cv2.imread('vishal.jpg')

orig_image = image.copy()
cv2.imshow('original image',orig_image)
cv2.waitKey(0)


gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
res,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)

_,contours,hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)


for c in contours:
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(orig_image,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.imshow('Bounding rectangle',orig_image)
    
cv2.waitKey(0)

for c in contours:
    accuracy = 0.0001 * cv2.arcLength(c,True)
    approx = cv2.approxPolyDP(c,accuracy,True)
    cv2.drawContours(image,[approx],0,(0,255,0),2)
    cv2.imshow('approx Poly DP',image)
    
cv2.waitKey(0)
cv2.destroyAllWindows()



















































