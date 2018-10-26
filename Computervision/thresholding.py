#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 08:52:46 2018

@author: ayush
"""


import cv2 as cv
import numpy as np

image = cv.imread('dog.jpg',0)
cv.imshow('Original',image)

#VALUES BELOW 127 GOES TO 0
ret,thresh1 = cv.threshhold(image,127,255,cv.THRESH_BINARY)
cv.imshow('1threshold binary',thresh1)



ret,thresh2 = cv.threshhold(image,127,255,cv.THRESH_BINARY_INV)
cv.imshow('threshold binary',thresh2)

ret,thresh3 = cv.threshhold(image,127,255,cv.THRESH_TRUNC)
cv.imshow('Thresh trunc',thresh3)

ret,thresh4 = cv.threshhold(image,127,255,cv.THRESH_TOZERO)
cv.imshow('thresh to zero',thresh4)

ret,thresh5 = cv.threshhold(image,127,255,cv.THRESH_TOZERO_INV)
cv.imshow('thresh to hold zero inv',thresh5)


cv.waitKey(0)
cv.destroyAllWindows()




#better way of adaptive thresholding





























