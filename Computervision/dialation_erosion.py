#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 09:48:22 2018

@author: ayush
"""
import cv2 as cv
import numpy as np


image = cv.imread('dog.jpg')
cv.imshow('Original',image)
cv.waitKey(0)

#DEFINE KERNEL SIZE
kernel = np.ones((5,5),np.uint8)

#NOW WE ERODE
erosion = cv.erode(image,kernel,iterations =1)
cv.imshow('erosion',erosion)
cv.waitKey(0)


dilation= cv.dilate(image,kernel,iterations =1)
cv.imshow('dilation',dilation)
cv.waitKey(0)





opening= cv.morphologyEx(image,cv.MORPH_OPEN,kernel)
cv.imshow('opening',opening)
cv.waitKey(0)



closing = cv.morphologyEx(image,cv.MORPH_CLOSE,kernel)
cv.imshow('closing',closing)
cv.waitKey(0)

cv.destroyAllWindows()







































































































































































