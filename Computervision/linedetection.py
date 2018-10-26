#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 10:25:55 2018

@author: ayush
"""
import cv2
import numpy as np


image = cv2.imread('chess.jpg')


gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,100,170,apertureSize = 3)

lines = cv2.HoughLines(edges,1,np.pi/180,150)

for rho,theta in lines[20]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(image,(x1,y1),(x2,y2),(255,0,0),2)

cv2.imshow('Hough lines',image)
cv2.waitKey(0)
cv2.destroyAllWindows()



















































