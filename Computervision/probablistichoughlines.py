#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 10:37:51 2018

@author: ayush
"""
import cv2
import numpy as np


image = cv2.imread('chess.jpg')


gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,100,170,apertureSize = 3)

lines = cv2.HoughLines(edges,1,np.pi/180,100,5,10)
print(lines.shape)
for x in range(0, len(lines)):
    
    for x1,y1,x2,y2 in lines[x]:###
        cv2.line(image,(x1,y1),(x2,y2),(0,255,0),3)

cv2.imshow('probablistic hough lines',image)
cv2.waitKey(0)
cv2.destroyAllWindows()













































