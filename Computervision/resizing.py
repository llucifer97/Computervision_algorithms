#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 20:31:49 2018

@author: ayush
"""
import cv2 as cv
import numpy as np
image = cv.imread('dog.jpg')
#let make our image 3/4 of it's original size
image_scaled = cv.resize(image,None,fx =0.75,fy = 0.75)
cv.imshow('Scaling-Linear Interpolation',image_scaled)
cv.waitKey()

#LETS DOUBLE THE SIZE OF OUR IMAGE
img_scaled = cv.resize(image,None,fx =2,fy = 2,interpolation = cv.INTER_CUBIC)
cv.imshow('Scaling-cubic Interpolation',img_scaled)
cv.waitKey()


#res ize by setting exact dimension
imG_scaled = cv.resize(image,(900,400),interpolation = cv.INTER_AREA)
cv.imshow('Scaling-SKWED SIZE',img_scaled)
cv.waitKey()
cv.destroyAllWindows()
