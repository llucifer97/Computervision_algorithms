#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 10:39:40 2018

@author: ayush
"""
import cv2
import numpy as np

image = cv2.imread('sudoku.jpg')
cv2.imshow('input image',image)
cv2.waitKey(0)

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

edged = cv2.Canny(gray,30,200)
cv2.imshow('Canny edges',edged)
cv2.waitKey(0)


_,contours,hierarchy = cv2.findContours(edged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
cv2.waitKey(0)

print("numbers ofcountours found = " + str(len(contours)))


cv2.drawContours(image,contours,-1,(0,255,0),3)
cv2.imshow('Contours',image)
cv2.waitKey(0)
cv2.destroyAllWindows()



























































































