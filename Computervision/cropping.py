#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 21:58:11 2018

@author: ayush
"""

import cv2 as cv
import numpy as np

image = cv.imread('sudoku.jpg')
height,width = image.shape[:2]

#let's get the starting pixels coordinates (TOP LEFT OF CROPPING RECTANGLE)
start_row , start_col = int(height*.25),int(width* .25)

end_row , end_col = int(height*.75),int(width* .75)

cropped = image[start_row:end_row, start_col:end_col]

cv.imshow("Original Image",image)
cv.waitKey(0)
cv.imshow("cropped Image",cropped)
cv.waitKey(0)
cv.destroyAllWindows()



























