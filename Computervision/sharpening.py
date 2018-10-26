#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 09:01:55 2018

@author: ayush
"""

import cv2 as cv
import numpy as np

image = cv.imread('dog.jpg')
cv.imshow('Original',image)

#CREATE OUR SHARPENING KERNWL,WE DON'T NORMALIZE SINCE THE VALUES IN MATRIX SUM TO 1
kernel_sharpening = np.array([[-1,-1,-1],
                             [-1,9,-1],
                             [-1,-1,-1]])


#APPLYING DIFFERENT KERNEL TO OUTPUT IMAGE
sharpened = cv.filter2D(image,-1,kernel_sharpening)
cv.imshow('image_sharening',sharpened)

cv.waitKey(0)
cv.destroyAllWindows()










































