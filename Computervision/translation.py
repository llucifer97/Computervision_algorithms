#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 19:59:31 2018
TRANSLATION
@author: ayush
"""
import cv2 as cv 
import numpy as np

image = cv.imread('dog.jpg')
#Store height and weight of image
height ,width = image.shape[:2]

quarter_height,quater_width = height/4, width/4
#          /1 0 Tx/
#          /0 1 Ty/
#T is our translational matrix
T = np.float32([[1,0,quater_width],[0,1,quarter_height]])
#WE USE WARPAFFINE TO TRANSFORM THE IMAGE USING THE MATRIX,T
img_translation = cv.warpAffine(image,T,(width,height))
cv.imshow('Translation',img_translation)
cv.waitKey()
cv.destroyAllWindows()


print(T)























