'''
https://github.com/georgesung/road_lane_line_detection
https://github.com/georgesung/advanced_lane_detection
https://medium.com/geekculture/4-techniques-self-driving-cars-can-use-to-find-lanes-fcb6dd06b633
https://towardsdatascience.com/tutorial-build-a-lane-detector-679fd8953132#127c

Start NEATO command: 
ros2 launch neato_node2 bringup.py host:=192.168.16.50

TODO: 
- Currently, the mask segment area is too small and the program breaks if it doesn't detect two lines. Fix this by either altering the mask 
- Add twist controls that corresponds the lane detection
    - Turn to keep neato in the center of the lane.
- Detect horizontal lines, stop and turn when a certain distance away. 
'''
import cv2
import matplotlib.pyplot as plt
import math
import os
import time
import numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge
from threading import Thread
from sensor_msgs.msg import Image
from copy import deepcopy
from geometry_msgs.msg import Twist, Vector3
from scipy.stats import linregress
import rclpy
from helper_functions import Point, Line, HoughLineDetection

class Lane_Detector(Node):
    """ Finds the lanes in the image. """

    def __init__(self, image_topic):
        super().__init__('lane_detector')
        self.cv_image = None                        # the latest image from the camera
        self.img_shape = None
        self.bridge = CvBridge()                    # used to convert ROS messages to OpenCV
        self.hough = HoughLineDetection()

        # drive
        self.twt = Twist()
        # rotation stuff
        self.start_orientation = None
        self.rot_speed = 0.3
        self.lin_speed = 0.1
        
        self.reset_lines_detected()
        self.calibrate_mask = True

        self.create_subscription(Image, image_topic, self.process_image, 10)
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)
        thread = Thread(target=self.loop_wrapper)
        thread.start()

    def reset_lines_detected(self):
        # Left and right lines of the lane
        self.left = None
        self.right = None
        self.horizontal = None
        self.lane_center_pt = None
        self.lane_center_line = None
        
    def process_image(self, msg):
        """ Process image messages from ROS and stash them in an attribute
            called cv_image for subsequent processing """
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        # self.img_shape = self.cv_image.shape
        self.reset_lines_detected()

    def get_Odom(self, msg):
        self.position = msg.pose.pose.position
        self.orientation = helper_functions.euler_from_quaternion(msg.pose.pose.orientation)

    def calc_lane_lines(self, lines):
        """ 
    
        """
        # Empty arrays to store the coordinates of the left and right lines
        left, right, horizontal = [], [], []

        # Loops through every detected line
        for line in lines:
            # Reshapes line from 2D array to 1D array
            x1, y1, x2, y2 = line.reshape(4)
            line = Line(x1, y1, x2, y2)
            # If slope is negative, the line is to the left of the lane, and otherwise, the line is to the right of the lane
            # TODO: add check here to make sure the line is not horizontal. We want to deal with that edge case separately. 
            if line.is_horizontal():
                horizontal.append((line.slope, line.y_intercept))
            elif line.slope < 0:
                left.append((line.slope, line.y_intercept))
            else:
                right.append((line.slope, line.y_intercept))
        # Averages out all the values for left and right into a single slope and y-intercept value for each line
        left_avg = np.average(left, axis = 0)
        right_avg = np.average(right, axis = 0)
        horizontal_avg = np.average(horizontal, axis = 0)
        # Caculate left and right lines for the lanes
        self.left = self.line_from_params(left_avg)
        self.right = self.line_from_params(right_avg)
        self.horizontal = self.line_from_params(horizontal_avg)

    def calc_lane_intersection(self):
        ''' Given the left and right lanes, figure out where the center of the lanes is.
        '''
        xdiff = (self.left.arr[0][0] - self.left.arr[1][0], self.right.arr[0][0] - self.right.arr[1][0])
        ydiff = (self.left.arr[0][1] - self.left.arr[1][1], self.right.arr[0][1] - self.right.arr[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            return None

        d = (det(*self.left.arr), det(*self.right.arr))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return Point(x,y)

    def calc_road_center(self):
        # Intersection point of the two lanes 
        self.lane_center_pt = self.calc_lane_intersection()
        
        avg_slope = np.average(self.left.slope + self.right.slope)
        avg_intercept = np.average(self.left.y_intercept + self.right.y_intercept)
        # Calculate center line
        self.lane_center_line = self.line_from_params((avg_slope, avg_intercept), self.lane_center_pt)

    def line_from_params(self, params:list, pt:Point = None) -> Line:
        ''' Calculate the two points that make up a line segment from the inputs. If just the default params are given, the line is calculated with hard coded y values that are convenient for visualization purposes. If a point on the line is given, a short line segment centered around the point will be retruned. 

            params: slope, y intercept
            pt: point the line passes through
        '''
        if params == None:
            return None
        m, b = params
        if pt != None:
            # line length
            len = 20
            # Point slope form: y-y1 = m(x-x1) -> (y-pt.y)/m + pt.x
            y1 = pt.y + len/2
            y2 = pt.y - len/2
            x1 = (y1 - pt.y)/m + pt.x
            x2 = (y2 - pt.y)/m + pt.x
        else: 
            # Sets initial y-coord as the the bottom of the frame and 500 above the bottom of the frame.
            y1 = self.img_shape[0]
            y2 = int(y1 - 500)
            # Slope intercept form: y=mx+b -> x=(y-b)/m
            x1 = int((y1 - b) / m)
            x2 = int((y2 - b) / m)
        return Line(int(x1), int(y1), int(x2), int(y2))

    def visualize_lanes(self):
        if self.left:
            self.left.draw(self.cv_image)
        if self.right:
            self.right.draw(self.cv_image)
        if self.lane_center_line:
            cv2.circle(self.cv_image, self.lane_center_pt.xy, radius=10, color=(255, 0, 0), thickness=-1)
            self.lane_center_line.draw(self.cv_image) 
    
    def display_coordinates_upon_click(self, event, x, y, flags, params):
        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            # displaying the coordinates on the Shell
            print(x, ' ', y)
            # displaying the coordinates on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.cv_image, str(x) + ',' +
                        str(y), (x,y), font,
                        1, (255, 0, 0), 2)
            cv2.imshow('video_window', self.cv_image)

    def approaching_horizontal_border(self):
        """ Use the horizontal line to inform whether we need to turn or not. 
        """
        threshold = self.img_shape[0]/2
        # Get the point on the line at the x center of the img frame. 
        x = self.img_shape[1]/2
        pt = self.horizontal.get_point_at_x(x)
        # If the point y is below the threshold, the line is too close and we need to turn. 
        if pt.y > threshold:
            # turn_dir = self.get_turn_direction()
            turn_dir = "left"
            self.turn_ninety_deg(turn_dir)
            # TODO: 90 degree turn of robot
            print("NEED TO TURN HERE")

    def get_turn_direction(self):
        pass
    
    # TODO: make a utils.py file and pull from there instead.
    def turn_ninety_deg(self, dir: str = "left"):
        """ Turn the neato 90 degrees left or right based on the direction of the speed. 
        """
        self.twt.linear = Vector3(x=0.0, y=0.0, z=0.0)
        # set rotation speed 
        if abs(self.start_orientation.z - self.orientation.z) >= math.pi/2:
            self.turning_flag = 0
            self.start_orientation = None
            self.twt.angular = Vector3(x=0.0, y=0.0, z=0.0)
        else: 
            self.twt.angular = Vector3(x=0.0, y=0.0, z=self.rot_speed)

    def drive_straight(self):
        """
        """
        self.twt.linear = Vector3(x=self.lin_speed, y=0.0, z=0.0)
        self.twt.angular = Vector3(x=0.0, y=0.0, z=0.0)

    def run_lane_detector(self):
        lines = self.hough.do_hough_line_transform(self.cv_image)
        self.calc_lane_lines(lines)
        # only find intersection if we see two lines lol
        if self.left and self.right:
            self.calc_road_center()
            print("center_pt", self.lane_center_pt)
            print(self.lane_center_line.slope)
            print(self.lane_center_line)
        self.visualize_lanes()

    def drive(self):
        """ This function determines how the robot will react and drive.        
        """
        if self.approaching_horizontal_border():
            self.turn_ninety_deg()
        else:
            self.drive_straight()
            
    def loop_wrapper(self):
        """ This function takes care of calling the run_loop function repeatedly.
            We are using a separate thread to run the loop_wrapper to work around
            issues with single threaded executors in ROS2 """
        cv2.namedWindow('video_window', 0)
        cv2.resizeWindow('video_window', 800, 500)
        while True:
            self.run_lane_detector()
            self.turn_ninety_deg()
            self.run_loop()
            time.sleep(0.1)

    def run_loop(self):
        # NOTE: only do cv2.imshow and cv2.waitKey in this function 
        if not self.cv_image is None:
            cv2.imshow('video_window', self.cv_image)
            cv2.waitKey(5)

            if self.calibrate_mask:
                cv2.setMouseCallback('video_window', self.display_coordinates_upon_click)
                cv2.waitKey(0)

"""
if __name__ == '__main__':
    img_path = "/home/jackie/ros2_ws/images/road_wide_angle_11_29_1669749490.4664428.png"
    img = cv2.imread(img_path)
    gray, canny = do_canny(img)
    segment =  do_segment(canny)
    houghlines = hough_transform(segment)
    left, right, middle_lane = calculate_lanes(img, houghlines)
    int_pt = calculate_lane_intersection(left, right)
    # print("intersection point:", int_pt, slope, intercept)
    # calculate midpoint line 
    # middle_line = calculate_coordinates(img, [slope, intercept])
    # print("middle line: ", middle_line)
    # middle_line.draw(gray)

    # cv2.imshow("segment", segment)
    # draw_hough_lines(houghlines, gray)
    left.draw(gray)
    right.draw(gray)
    middle_lane.draw(gray)
    cv2.circle(gray, (int(int_pt.x), int(int_pt.y)), radius=10, color=(255, 0, 0), thickness=-1)


    cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", gray)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
"""

# if __name__ == '__main__':
#     node = Lane_Detector("/camera/image_raw")
#     node.run()

# def main(args=None):
#     rclpy.init()
#     n = Lane_Detector("camera/image_raw")
#     rclpy.spin(n)
#     rclpy.shutdown()


# if __name__ == '__main__':
#     main()

def main(args=None):
    rclpy.init(args=args)      # Initialize communication with ROS
    node = Lane_Detector("camera/image_raw")   # Create our Node
    rclpy.spin(node)           # Run the Node until ready to shutdown
    rclpy.shutdown()           # cleanup

if __name__ == '__main__':
    main()