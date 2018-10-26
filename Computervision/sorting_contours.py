#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 00:02:07 2018

@author: ayush
"""
import cv2
import numpy as np

image = cv2.imread('sudoku.jpg')
cv2.imshow('0 - Original Image',image)
cv2.waitKey(0)

blank_image = np.zeros((image.shape[0],image.shape[1],3))

original_image = image
gray = cv2.cvtColor(image,0)

edged = cv2.Canny(gray,50,200)

cv2.imshow('1 - canny Edges',edged)
cv2.waitKey(0)

_,contours, hierarchy = cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE )
print("Numbers of countours found = ",len(contours))

cv2.drawContours(blank_image,contours,-1,(0,255,0),3)
cv2.imshow('2 - All contours over blank image',blank_image)
cv2.waitKey(0)

cv2.drawContours(image,contours,-1,(0,255,0),3)
cv2.imshow('3 - AllContours',image)
cv2.waitKey(0)

cv2.destroyAllWindows()
shape = image.shape

def get_contours_areas(contours):
    all_areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        all_areas.append(area)
    return all_areas


cnt = contours[4]
cv2.drawContours(image, [cnt], 0, (0,255,0), 3)


#cv2.drawContours(image,contours,700,(0,255,0),3)
cv2.imshow('Contours',image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#image =cv2.imread('sudoku.jpg')
original_image = image

print('contours area before sorting')
print(get_contours_areas(contours))



sorted_contours = sorted(contours, key = cv2.contourArea , reverse = True)

print('contours area after sorting')
print(get_contours_areas(sorted_contours))



#for c in sorted_contours:
#    cv2.drawContours(original_image,[c],-1,(255,0,0),3)
#    #cv2.waitKey(0)
#    cv2.imshow('contours by area', original_image)
#    
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#selected the largest contour
cnt = sorted_contours[0]
cv2.drawContours(image, [cnt], 0, (0,255,0), 3)
cv2.imshow('Contours',image)
cv2.waitKey(0)
cv2.destroyAllWindows()



idx = 0 # The index of the contour that surrounds your object
mask = np.zeros_like(image) # Create mask where white is what we want, black otherwise
cv2.drawContours(mask, [cnt], 0, 255, -1) # Draw filled contour in mask

out = np.zeros_like(image) # Extract out the object and place into output image
out[mask == 255] = image[mask == 255]

# Now crop
(x, y,z) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
out = out[topx:bottomx+1, topy:bottomy+1]

# Show the output image
cv2.imshow('Output', out)
cv2.waitKey(0)
cv2.destroyAllWindows()


gray1 = cv2.cvtColor(out,0)


lines = cv2.HoughLines(gray1,1,np.pi/180,240)

for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * np.cos(theta)
    y0 = b * np.sin(theta)
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(image,(x1,y1),(x2,y2),(255,0,0),2)
    
cv2.imshow('HoughLines',out)
cv2.waitKey(0)
cv2.destroyAllWindows()




























