#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 22:08:08 2018

@author: ayush
"""
#
#import cv2 as cv
#import numpy as np
#
#image = cv.imread('vishal.jpg')
#
##CREATE A MATRIX OF ONES,THEN MULTIPLY IT BY A SCALER OF 100
##THIS GIVES A MATRIX WITH SOME DIMENSIONS OF OUR IMAGE WITH ALL VALUES BEING 100
#M = np.ones(image.shape,dtype ="uint8")*75
#
##WE USE THIS TO ADD THIS MATRIX M,TO OUR LARGE NOTICE THR INCRESES IN BRIGHTNESS
#added = cv.add(image,M)
#cv.imshow("Added",added)
#
##LIKEWISE WE CAN ALSO SUBTRACT NOTICE THE DECRESES IN BRIGHTNESS
#subtracted = cv.subtract(image,M)
#cv.imshow("subtracted",subtracted)
#
#cv.waitKey(0)
#cv.destroyAllWindows

#BITWISE OPERATIO INTRODUCTION

import cv2 as cv
import numpy as np

image = cv.imread('vishal.jpg')
#MAKING A SQURE
square = np.zeros((300,300), np.uint8)
cv.rectangle(square,(50,50),(250,250),255,-2)
cv.imshow("Square",square)
cv.waitKey(0)

#MAKE A ELLIPSE
ellipse = np.zeros((300,300),np.uint8)
cv.ellipse(ellipse,(150,150),(150,150),30,0,180,255,-1)
cv.imshow("ellipse",ellipse)
cv.waitKey(0)



#SHOW ONLY WHEN THEY INTERSECT
And  = cv.bitwise_and(square,ellipse)
cv.imshow("AND",And)
cv.waitKey(0)


#SHOW  WHEN THEY INTERSECT
bitwiseOr  = cv.bitwise_or(square,ellipse)
cv.imshow("AND",And)
cv.waitKey(0)


#SHOW ONLY WHEN THEY INTERSECT
bitwiseXor  = cv.bitwise_xor(square,ellipse)
cv.imshow("AND",And)
cv.waitKey(0)

#SHOW ONLY WHEN THEY INTERSECT
bitwiseNot_sq  = cv.bitwise_not(square,ellipse)
cv.imshow("AND",And)
cv.waitKey(0)


cv.destroyAllWindows()
















































































