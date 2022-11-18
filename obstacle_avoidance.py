import cv2 as cv
import numpy as np
import os

def do_canny(frame):
    # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    # Applies a 5x5 gaussian blur with deviation of 0 to frame - not mandatory since Canny will do this for us
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    # Applies Canny edge detector with minVal of 50 and maxVal of 150
    canny = cv.Canny(blur, 50, 150)
    return canny

def do_segment(frame, x, y, w, h):
    # Since an image is a multi-directional array containing the relative intensities of each pixel in the image, we can use frame.shape to return a tuple: [number of rows, number of columns, number of channels] of the dimensions of the frame
    # frame.shape[0] give us the number of rows of pixels the frame has. Since height begins from 0 at the top, the y-coordinate of the bottom of the frame is its height
    height = frame.shape[0]
    # Creates a triangular polygon for the mask defined by three (x, y) coordinates
    polygons = np.array([
                            [(x, y), (x+w, y), ()]
                        ])
    # Creates an image filled with zero intensities with the same dimensions as the frame
    mask = np.zeros_like(frame)
    # Allows the mask to be filled with values of 1 and the other areas to be filled with values of 0
    cv.fillPoly(mask, polygons, 255)
    # A bitwise and operation between the mask and frame keeps only the triangular area of the frame
    segment = cv.bitwise_and(frame, mask)
    return segment

def calculate_lines(frame, lines):
    # Empty arrays to store the coordinates of the left and right lines
    left = []
    right = []
    # Loops through every detected line
    for line in lines:
        # Reshapes line from 2D array to 1D array
        x1, y1, x2, y2 = line.reshape(4)
        # Fits a linear polynomial to the x and y coordinates and returns a vector of coefficients which describe the slope and y-intercept
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        y_intercept = parameters[1]
        # If slope is negative, the line is to the left of the lane, and otherwise, the line is to the right of the lane
        if slope < 0:
            left.append((slope, y_intercept))
        else:
            right.append((slope, y_intercept))
    # Averages out all the values for left and right into a single slope and y-intercept value for each line
    left_avg = np.average(left, axis = 0)
    right_avg = np.average(right, axis = 0)
    # Calculates the x1, y1, x2, y2 coordinates for the left and right lines
    left_line = calculate_coordinates(frame, left_avg)
    right_line = calculate_coordinates(frame, right_avg)
    return np.array([left_line, right_line])

def calculate_coordinates(frame, parameters):
    slope, intercept = parameters
    # Sets initial y-coordinate as height from top down (bottom of the frame)
    y1 = frame.shape[0]
    # Sets final y-coordinate as 150 above the bottom of the frame
    y2 = int(y1 - 150)
    # Sets initial x-coordinate as (y1 - b) / m since y1 = mx1 + b
    x1 = int((y1 - intercept) / slope)
    # Sets final x-coordinate as (y2 - b) / m since y2 = mx2 + b
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def visualize_lines(frame, lines):
    # Creates an image filled with zero intensities with the same dimensions as the frame
    lines_visualize = np.zeros_like(frame)
    # Checks if any lines are detected
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            # Draws lines between two coordinates with green color and 5 thickness
            cv.line(lines_visualize, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return lines_visualize

directory = "/home/simrun/ros2_ws/src/comprobo_road_navigation/sample_images/right/"

def main():
    # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
    for image in os.listdir(directory):
        # Using cv2.imread() method
        frame = cv.imread(directory + image )
        canny = do_canny(frame)
        # cv.imshow("canny", canny)
        

    # idea options:
    # try out a color mask and rectangle detection

        # convert image to grayscale image
        gray_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # convert the grayscale image to binary image
        ret,thresh = cv.threshold(gray_image,127,255,0)
        
        # calculate moments of binary image
        M = cv.moments(thresh)
        contours,hierarchy = cv.findContours(thresh, 1, 2)
        filtered_contours = []
        areas = []
        for contour in contours:
            area = cv.contourArea(contour)
            areas.append(area)
            if area > 100 and area < 2000:
                filtered_contours.append(contour)

        mask = np.zeros(frame.shape[:2], dtype="uint8")
        max_value = max(areas)
        largest_contour = contours[areas.index(max_value)]
        x,y,w,h = cv.boundingRect(largest_contour)
        cv.rectangle(mask,(x,y),(x+w,y+h), 255, -1)
        masked = cv.bitwise_and(frame, frame, mask=mask)
        cv.imshow("Mask Applied to Image", masked)

        # # cv.imshow("thres", thresh)
        # cv.drawContours(frame, contours, -1, (0, 255, 0), 1)
        # cv.imshow("contours", frame);
        cv.waitKey(0)
        # create a custom mask for left and right and use that
    cv.destroyAllWindows()


if __name__ == "__main__":
   main()