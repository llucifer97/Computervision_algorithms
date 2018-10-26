#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 21:51:08 2018

@author: ayush
"""

import cv2 as cv

image = cv.imread('vishal.jpg')

smaller = cv.pyrDown(image)
larger = cv.pyrUp(smaller)

cv.imshow('Original',image)
cv.imshow('Smaller',smaller)
cv.imshow('Larger',larger)
cv.waitKey(0)
cv.destroyAllWindows()

























