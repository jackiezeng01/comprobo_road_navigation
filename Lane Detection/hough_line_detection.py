'''
https://github.com/georgesung/road_lane_line_detection
https://medium.com/geekculture/4-techniques-self-driving-cars-can-use-to-find-lanes-fcb6dd06b633
https://towardsdatascience.com/tutorial-build-a-lane-detector-679fd8953132#127c

'''
import cv2
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import math
import os

# Global parameters:


# Applying canny detector
def canny_detector():
    # noise reduction
    # apply a 5x5 Gaussian filter to 
    
    # intensity gradient
    # apply a sobel, roberts, or prewitt kernel

    # non-maximum suppression applied to sharpen the edges.

def gaussian_blur(img, kernel_size):
    """Apply the gaussian noise kernel to smooth the image."""
    blur = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    return blur

def 
if __name__ == '__main__':
    