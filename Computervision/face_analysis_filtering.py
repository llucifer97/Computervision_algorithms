#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 15:50:08 2018

@author: ayush
"""
import cv2
import dlib
import numpy as np

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()



def annotate_landmarks(im,landmarks):
    im =im.copy()
    for idx,point in enumerate(landmarks):
        pos = (point[0,0],point[0,1])
        cv2.putText(im,str(idx),pos,
                    frontFace = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale = 0.4,
                    color = (0,0,255))
        cv2.circle(im, pos, 3, color = (0, 255, 255))
        
    return im

image = cv2.imread('ayush.jpg')
landmarks = get_landmarks(image)

image_with_landmarks = annotate_landmarks(image,landmarks)

cv2.imshow('Result',image_with_landmarks)
cv2.imwrite('image_with_landmarks.jpg',image_with_landmarks)
cv2.waitKey(0)
cv2.destroyAllWindows()






























