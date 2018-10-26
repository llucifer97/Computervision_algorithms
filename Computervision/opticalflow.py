#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 16:46:21 2018

@author: ayush
"""


import numpy as np
import cv2

cap = cv2.VideoCapture('slow.flv')

feature_params = dict(maxCorners = 100,
                      qualityLevel = 0.3,
                      minDistance = 7,
                      blockSize = 7)

lucas_kanade_params = dict(winSize = (15,15),
                           maxLevel = 2,
                           criteria  = (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,10,0.03))



color = np.random.randint(0,255,(100,3))

ret,prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame,cv2.COLOR_BGR2GRAY)

prev_corners = cv2.goodFeaturesToTrack(prev_gray,mask = None,**feature_params)

mask = np.zeros_like(prev_frame)

while(1):
    
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    new_corners,status,errors = cv2.calcOpticalFlowPyrLK(prev_gray,frame_gray,prev_corners,None,**lucas_kanade_params)
    
    good_new = new_corners[status==1]
    good_old = prev_corners[status == 1]
    
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        
        a,b = new.ravel()
        c,d = old.ravel()
        
        mask = cv2.line(mask,(a,b),(c,d),color[i].tolist(),2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
        
    img = cv2.add(frame,mask)
    
    
    cv2.imshow('optical flow lucas kanade',img)
    
    if cv2.waitKey(1) == 13:
        break
    
    prev_gray = frame_gray.copy()
    prev_corners = good_new.reshape(-1,1,2)
    
    
cv2.destroyAllWindows()
cap.release()
     
        























































