'''
image to use:
/home/jackie/ros2_ws/src/comprobo_road_navigation/sample_images/right/road_11_13_1668387082.230232.png

Advanced lane detection: 
https://github.com/georgesung/advanced_lane_detection

Hough line transforms:
https://github.com/georgesung/road_lane_line_detection

'''

# Undistort input image with camera calibration

# Thresholded binary image
'''
Apply a bunch of binary image filters and combine them
'''

# Perform a perspective transform
'''
We can get a bird's eye view of the lane, which allows us to fit a curved line to the lane lines. 
'''

# Create polynomial fit
'''
Take the warped binary image and fit a 2nd order polynomal on the left and right lanes. 
'''

# Calculate the radius of curvature for left and right lanes

# Get vehicle offset from lane center