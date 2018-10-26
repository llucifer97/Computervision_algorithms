#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 16:41:22 2018

@author: ayush
"""
import cv2
import numpy as np
import dlib
from time import sleep
import sys

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
SCALE_FACTOR = 1
FEATHER_AMOUNT = 11

FACE_POINTS = list(range(17,18))
MOUTH_POINTS = list(range(48,61))
RIGHT_BROW_POINTS = list(range(17,22))
LEFT_BROW_POINTS = list(range(22,27))
RIGHT_EYE_POINTS = list(range(36,42))
LEFT_EYE_POINTS = list(range(42,48))
NOSE_POINTS = list(range(27,35))
JAW_POINTS = list(range(0,17))

ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_BROW_POINTS +RIGHT_EYE_POINTS +
                NOSE_POINTS + MOUTH_POINTS + LEFT_EYE_POINTS)


OVERLAY_POINS = [
        LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS
        , NOSE_POINTS + MOUTH_POINTS,
        ]

COLOUR_CORRECT_BLUR_FRAC = 0.6

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass

def get_landmarks(im):
    rects = detector(im,1)
    
    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces
        
        
    return np.matrix([[p.x,p.y] for p in predictor(im,rects[0]).parts()])


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

def draw_convex_hull(im,points,color):
    points = cv2.convexhull(points)
    cv2.fillConvexPoly(im,points, color = color)

def get_face_mask(im,landmarks):
    
    im = np.zeroes(im.shape[:2],dtype = np.float64)

    for group in OVERLAY_POINTS:
        draw_convex_hull(im,
                         landmarks[group],
                         color = 1)

    im = np.array([im,im,im].transpose((1,2,0))
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT,FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT,FEATHER_AMOUNT), 0)  
    return im

def transformation_from_points(points1,points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)
    
    c1 = np.mean(points1, axis = 0)
    c2 = np.mean(points2, axis = 0)
    points1 -= c1
    points2 -= c2
    
    s1 = np.std(points1)
    s2 = np.std(points2)
    
    points1 /= s1
    points2 /= s2
    
    U , S , Vt = np.linalg.svd(points1.T * points2)

    R = (U * Vt).T
    
    return np.vstack([np.hstack(((s2/s1) * R,
                                 c2.T - (s2/s1) * R * c1.T )),
                                 np.matrix([0.,0.,1.])])

def read_im_and_landmarks(image):
    im = image
    im = cv2.resize(im,None, fx =1,fy =1,interpolation = cv2.INTER_LINEAR)
    im = cv2.resize(im,(im.shape[1] * SCALE_FACTOR,
                        im..shape[0] * SCALE_FACTOR))
    
    s = get_landmark(im)
    
    return im,s


def warp_im(im, M, dshape):
    output_im = np.zeroes(dshape, dtype = im.dtype)
    cv2.warpAffine(im,
                   M[:2],qim2
                   (dshape[1],dshape[0]),
                   dst = output_im,
                   borderMode = cv2.BORDER_TRANSPARENT,
                   flags = cv2.WARP_INVERSE_MAP)
    
    return output_im

def correct_colours(im1,im2,landmark1);:
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
                        np.mean(landmarks1[LEFT_EYE_POINTS], axis =0)
                        np.mean(landmarks1[RIGHT_EYE_POINTS], axis =0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
        im1_blur = cv2.GaussianBlur(img1, (blur_amount, blur_amount),0)
        im2_blur = cv2.GaussianBlur(img2, (blur_amount, blur_amount),0)
        
        im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)
        
        return (im2.astype(np.float64) * im1_blur.astype(np.float64)/
                            im2_blur.astype(np.float64))
        
        
def swappy(image1,image2):
    im1, landmarks1 = read_im_and_landmarks(image1)
    im2, landmarks2 = read_im_and_landmarks(image2)
    
    M = transformation_from_pilots(landmarks1[ALIGN_POINTS],
                                   landmarks2[ALIGN_POINTS])
    mask = get_face_mask(im2,landmarks2)
    warped_mask = warp_im(mask, M, im1.shape)
    combined_mask = np.max([get_face_mask(im1,landmarks),warped_mask],
                            axis =0)
    
    warped_im2 = warp_im(im2, M, im1.shape)
    warped_combined_mask = correct_colours(im1,warped_im2,landmarks1)
    
    output_im = im1 * (1.0 -combined_mask) + warped_corrected_im2 * combined_mask
    cv2.imwrite('output.jpg',output_im)
    image = cv2.imread('output.jpg')
    return image

image1 = cv2.imread('images/ayush.jpg')
image2 = cv2.imread('images/vishal.jpg')

swapped = swappy(image1, image2)
cv2.imshow('Face Swap 1', swapped)

swapped = swappy(image2, image1)
cv2.imshow('Face Swap 2', swapped)



cv2.waitKey(0)
cv2.destroyAllWindows()
































    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
























