'''
https://github.com/georgesung/road_lane_line_detection
https://medium.com/geekculture/4-techniques-self-driving-cars-can-use-to-find-lanes-fcb6dd06b633
https://towardsdatascience.com/tutorial-build-a-lane-detector-679fd8953132#127c

'''
import cv2
import matplotlib.pyplot as plt
import math
import os

# Global parameters:


# Applying canny detector
def canny_detector(frame):
    # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # noise reduction
    # apply a 5x5 Gaussian filter to 
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # intensity gradient
    # apply a sobel, roberts, or prewitt kernel
    canny = cv2.Canny(blur, 50, 100)
    return canny
    # non-maximum suppression applied to sharpen the edges.

def gaussian_blur(img, kernel_size):
    """Apply the gaussian noise kernel to smooth the image."""
    blur = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    return blur

if __name__ == '__main__':
    img_path = "/home/jackie/ros2_ws/images/road_wide_angle_11_29_1669749491.8072128.png"
    img = cv2.imread(img_path)
    canny = canny_detector(img)
    cv2.namedWindow("image")
    cv2.imshow('image',canny)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
