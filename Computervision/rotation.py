#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 20:20:22 2018
ROTATION
@author: ayush
"""
import cv2 as cv 
import numpy as np

image = cv.imread('dog.jpg')
#Store height and weight of image
height ,width = image.shape[:2]

#DIVIDE  BY TWO TO ROTATE THE IMAGE AROUND ITSCENTRE
rotation_matrix= cv.getRotationMatrix2D((width/2,height/2),90,1)
rotated_image  = cv.warpAffine(image,rotation_matrix,(width,height))
cv.imshow('Rotated Image',rotated_image)
cv.waitKey(0)
cv.destroyAllWindows()

#OTHER OPTION TO ROTATE TO AVOID BLANK SPACES


img = cv.imread('dog.jpg')


rotatedimage = cv.transpose(img)
cv.imshow('Rotated Image method2',rotatedimage)
cv.waitKey()
cv.destroyAllWindows()








































































