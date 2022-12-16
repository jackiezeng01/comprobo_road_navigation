""" This file handles the lane following behavior. 

Resources referenced: 
https://github.com/georgesung/road_lane_line_detection
https://github.com/georgesung/advanced_lane_detection
https://medium.com/geekculture/4-techniques-self-driving-cars-can-use-to-find-lanes-fcb6dd06b633
https://towardsdatascience.com/tutorial-build-a-lane-detector-679fd8953132#127c

Start fisheye NEATO command: 
ros2 launch neato_node2 bringup.py host:=192.168.16.50

"""

import cv2
import time
import numpy as np
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist, Vector3
from comprobo_road_navigation.helper_functions import Point, Line, HoughLineDetection
import enum 
 
# Enums for the position of the robot in the lane
class Position_in_Lane(enum.Enum):
    centered = 1
    too_right = 2
    too_left = 3

class Lane_Follower():
    """ Finds the lanes in the image and calculate directions for"""

    def __init__(self):
        self.cv_image = None                        # the latest image from the camera
        self.img_shape = [668, 978]                 # image shape after undistorting and cropping
        self.bridge = CvBridge()                    # used to convert ROS messages to OpenCV
        self.hough = HoughLineDetection()           # class for line detection

        # 90 degree turn vars
        self.turning_flag = 0
        self.turn_start_time = None
        self.ninety_deg_turn_time = 5 #sec
        
        # Drive vars
        self.twt = Twist()
        self.rot_speed = None
        self.lin_speed = None
        
        # Intiate all detected lanes and calcuations to None
        self.reset()

        # Toggle to create a new polygon to mask off the road portion
        self.calibrate_mask = False

        # Slope of the horizontal line must between 0 and -/+ of this threshold
        self.horizontal_slope_threshold = 0.15
        # Range of slope values to be considered a left/right lane
        self.lane_slope_detection_range = [0.5, 2]   
        # Expected and acceptable range from the expected slope for the robot to be centered in the lane. 
        self.lane_slope_centered = [0.75, 0.05]      # [expected value, acceptable range from expected value]

        # If the horizontal line detected is greater than this y threshold, the 
        # robot is approaching an edge and should turn 90 degrees.
        self.horizontal_y_threshold = 425   
            
        # Saves the last detected lane - could be right/left
        self.last_lane_before_horizontal = None
        # Counter for the number of horizontal lines detected after a 90 degree turn. 
        # The robot will only turn if the number of horizontal liens surpass a 
        # theshold. This prevents accidental turns at detecting horizontal tile lines. 
        self.num_horizontal_lines_detected = 0

    def reset(self):
        """ Set the variables that should be reset with each new frame
        """
        self.left = None
        self.left_min_y = None
        self.right = None
        self.right_min_y = None
        self.horizontal = None
        self.lane_center_pt = None
        self.neato_position_in_lane = None
    
    def calc_lane_lines(self, lines:list):
        """ Calculate the left, right, and horizontal lane lines.

            Args: 
                lines: list of lines in the format (x1, y1, x2, y2) detected 
                through hough line transformation
        """
        # Store the the params (slope, y intercept) of the left and right lines
        left, right, horizontal = [], [], []
        # Store the y values of the endpoints of each line. This is used later 
        # to determine whether the lines are below the detected horizontal line.
        left_y, right_y = [], []

        # Loops through lines and group lines based on slopes into horizonal, 
        # left, and right.
        for line in lines:
            # Reshapes line from 2D array to a line object
            x1, y1, x2, y2 = line.reshape(4)
            line = Line(x1, y1, x2, y2)
            # Categorize lines based on slope
            if line.is_horizontal(self.horizontal_slope_threshold):
                horizontal.append((line.slope, line.y_intercept))
            elif line.is_within_lane_slope_detection_range(self.lane_slope_detection_range):
                if line.slope < 0:
                    left.append((line.slope, line.y_intercept))
                    left_y.extend((line.pt1.y,line.pt2.y))
                else:
                    right.append((line.slope, line.y_intercept))
                    right_y.extend((line.pt1.y,line.pt2.y))
        
        # Average the slope and y intercept in each group and calculate the
        # horizontal, left, and right lanes. 
        if horizontal != []:
            horizontal_avg = np.average(horizontal, axis = 0)
            self.horizontal = self.line_from_params(horizontal_avg)        
        if left != []:
            # Averages out all the values for left and right into a single slope and y-intercept value for each line
            left_avg = np.average(left, axis = 0)
            self.left_min_y = min(left_y)
            self.left = self.line_from_params(left_avg)
        if right != []:
            right_avg = np.average(right, axis = 0)
            self.right_min_y = min(right_y)
            self.right = self.line_from_params(right_avg)
        
        # Set the last lane detected
        if self.left and self.left_min_y:
            if self.left_min_y > self.horizontal_y_threshold-50:
                self.last_lane_before_horizontal = "left"
        elif self.right and self.right_min_y:
            if self.right_min_y > self.horizontal_y_threshold-50: 
                self.last_lane_before_horizontal = "right"

    def calc_lane_intersection(self):
        """ Calculate the intersection point of the left and right lanes.
        """
        # Calculate the intersection point of the two lanes 
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

        self.lane_center_pt = Point(x,y)

    def line_from_params(self, params:list, pt:Point = None) -> Line:
        ''' Calculate the two points that make up a line segment from the inputs.
            
            If just the default params are given, the line is calculated with 
            hard coded y values that are convenient for visualization purposes.
            If the slope is basically 0 (horizontal), create a horizontal line. 

            Returns:
                params: list of (slope, y intercept)
                pt: point the line passes through
        '''
        m, b = params
        # Horizontal line
        if (abs(m)<self.horizontal_slope_threshold):
            # if the slope is basically 0, so horizontal
            x1 = 0
            x2 = self.img_shape[1]
            y1, y2 = b, b
        # Line from m,b
        else: 
            # Sets initial y-coord as the the bottom of the frame and 500 above the bottom of the frame.
            y1 = self.img_shape[0]
            y2 = int(y1 - 500)
            # Slope intercept form: y=mx+b -> x=(y-b)/m
            x1 = int((y1 - b) / m)
            x2 = int((y2 - b) / m)
        return Line(int(x1), int(y1), int(x2), int(y2))

    def visualize_lanes(self):
        """ Visualize detected lanes and the calculated intersection point.
        """
        if self.left:
            self.left.draw(self.cv_image)
        if self.right:
            self.right.draw(self.cv_image)
        if self.horizontal is not None:
            self.horizontal.draw(self.cv_image)
        if self.lane_center_pt is not None:
            cv2.circle(self.cv_image, self.lane_center_pt.xy, radius=10, color=(255, 0, 0), thickness=-1)
    
    def update_lane_detection_area(self, event, x, y, flags, params):
        """ Outputs the lane detection 
        """
        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN and len(self.hough.polygon_pts)<6:
            # displaying the coordinates on the Shell
            print(x, ' ', y)
            self.hough.polygon_pts.append((x,y))
            print(self.hough.polygon_pts)
            print(len(self.hough.polygon_pts))
            # displaying the coordinates on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.cv_image, str(x) + ',' +
                        str(y), (x,y), font,
                        1, (255, 0, 0), 2)
            cv2.imshow('video_window', self.cv_image)

    def approaching_horizontal_border(self):
        """ Use the horizontal line to inform whether the robot needs to make a 
            90 turn. If the detected horizontal line is too close to the robot,
            the robot is approaching a border and should turn.

            Returns:
                Bool: Whether the robot is approaching a horizontal border
        """
        if self.horizontal is None:
            return False
        # Get the point on the line at the x center of the img frame. 
        x = self.img_shape[1]/2
        pt = self.horizontal.get_point_at_x(x)
        self.num_horizontal_lines_detected += 1
        # If the point y is below the threshold, the line is too close and we need to turn. 
        if pt.y > self.horizontal_y_threshold:
            return True

    def get_turn_direction(self):
        """ Get which direction the robot should turn when it approaches a 
            horizontal line. If the last lane detected was left, it most likely 
            means the left side is blocked by a lane and the robot should turn 
            left. Vice versa if it last detected a right lane. 
        
            Returns: 
                dir: "right"/"left" - direction the robot should turn 90 deg in
        """
        dir = None
        if self.last_lane_before_horizontal == "right":
            dir = "left"
        elif self.last_lane_before_horizontal == "left":
            dir = "right"
        return dir

    def evaluate_neato_position_in_lane(self):
        """ Evaluate neato's position in the lane lines. If two lane lines are 
            detected, this is evaluated based on the calculated intersection point. 
            If one lane line is detected, this is determined based on the slope 
            of the lane line.
        """
        expected_slope, threshold = self.lane_slope_centered

        # If the number of horizontal lines are consistently detected, the robot
        # should just proceed forward until it reaches the horizontal line. This
        # prevents the robot from following possibly distorted lane lines.
        if (self.num_horizontal_lines_detected > 5):
            pos = Position_in_Lane.centered

        # If two lanes are detected and there is an intersection point, evaluate
        # based on that.
        elif self.lane_center_pt:
            # Acceptable center threshold
            center_threshold = 20
            center_range = [self.img_shape[1]/2-center_threshold, 
                            self.img_shape[1]/2+center_threshold]
            # Evaluate based on the lane intersection point
            x = self.lane_center_pt.x
            if center_range[0] <= x <= center_range[1]:
                pos = Position_in_Lane.centered
            elif x < center_range[0]:
                pos = Position_in_Lane.too_right
            else:
                pos = Position_in_Lane.too_left
            
        # If there is only one lane detected, determine whether the robot is 
        # centered in the lane with the slope of the right/left lane.
        elif self.left:
            slope = abs(self.left.slope)
            if expected_slope-threshold < slope < expected_slope+threshold:
                pos = Position_in_Lane.centered
            if expected_slope-threshold >= slope:
                # print("left slope: ", self.left.slope)
                pos = Position_in_Lane.too_right
            elif slope >= expected_slope+threshold:
                # print("left slope: ", self.left.slope)
                pos = Position_in_Lane.too_left
        elif self.right:
            slope = abs(self.right.slope)
            if expected_slope-threshold < slope < expected_slope+threshold:
                pos = Position_in_Lane.centered
            if expected_slope-threshold >= slope:
                # print("right slope: ", self.right.slope)
                pos = Position_in_Lane.too_left
            elif slope >= expected_slope+threshold:
                # print("right slope: ", self.right.slope)
                pos = Position_in_Lane.too_right
        else: 
            pos = Position_in_Lane.centered
        
        # Update neato pos variable.
        self.neato_position_in_lane = pos

#
# Drive functions --------------------------------------------------------------
#
    def turn_ninety_deg(self):
        """ Turn the neato 90 degrees left or right based on the direction of 
            the speed. This function sets self.twt (the twist command)
        """
        # Set linear speed to 0 so the robot turns in place
        self.twt.linear = Vector3(x=0.0, y=0.0, z=0.0)

        if (self.turning_flag == 1):
            # Stop turning if the turn time has been reached
            if abs(self.turn_start_time - time.time()) >= self.ninety_deg_turn_time:
                self.turning_flag = 0
                self.twt.angular = Vector3(x=0.0, y=0.0, z=0.0)
                print("stop turn")
            
            # Turn based on calculated turn direction.
            elif(self.turn_dir == "left"):
                print("turning left")
                self.twt.angular = Vector3(x=0.0, y=0.0, z=self.rot_speed)
            elif(self.turn_dir == "right"):
                print("turning right")
                self.twt.angular = Vector3(x=0.0, y=0.0, z=-self.rot_speed)

    def drive_straight(self):
        """ Set Twist command to drive straight
        """
        self.twt.linear = Vector3(x=self.lin_speed, y=0.0, z=0.0)
        self.twt.angular = Vector3(x=0.0, y=0.0, z=0.0)
   
    def adjust_leftward(self):
        """ Set Twist to adjust leftward while still driving forward. Used for 
            correcting robot position in the lane.
        """
        self.twt.linear = Vector3(x=self.lin_speed, y=0.0, z=0.0)
        self.twt.angular = Vector3(x=0.0, y=0.0, z=self.rot_speed/2)
 
    def adjust_rightward(self):
        """ Set Twist to adjust rightward while still driving forward. Used for 
            correcting robot position in the lane.
        """
        self.twt.linear = Vector3(x=self.lin_speed, y=0.0, z=0.0)
        self.twt.angular = Vector3(x=0.0, y=0.0, z=-self.rot_speed/2)

    def drive_within_the_lane(self):
        """ Drive the robot within the lane based on the determined neato position. 
        """
        if self.neato_position_in_lane == Position_in_Lane.centered:
            self.drive_straight()
        elif self.neato_position_in_lane == Position_in_Lane.too_right:
            self.adjust_leftward()
        elif self.neato_position_in_lane == Position_in_Lane.too_left:
            self.adjust_rightward()
        else:
            self.drive_straight()

    def drive(self):
        """ This function determines how the robot will react and drive.        
        """
        # Start turn behavior if approaching horizontal border
        if self.approaching_horizontal_border():
            # print("TURN HERE")
            self.turn_dir = self.get_turn_direction()
            self.turning_flag = 1
            self.turn_start_time = time.time()          # Start time of the turn
            self.num_horizontal_lines_detected = 0
        # Turn behavior
        if self.turning_flag:
            self.turn_ninety_deg()
        # Lane follow
        else: 
            self.evaluate_neato_position_in_lane()
            self.drive_within_the_lane()
        # Twist command
        return self.twt

#
# Main functions --------------------------------------------------------------
#
    def run_lane_follower(self, image, lin_speed, rot_speed, orientation, position):
        """ Lane follower main function. Runs lane followuing behavior

            Inputs:
                image: undistorted cv image
                lin_speed: linear speed of the Neato
                rot_speed: rotational speed of the Neato
                orientation: current odom orientation
                position: current odom position

            Returns:
                self.drive(): Twist command for how to move the Neato
                self.cv_image: Image with visualiztions added.

        """
        self.cv_image = image
        self.lin_speed = lin_speed
        self.rot_speed = rot_speed
        self.orientation = orientation
        self.position = position
        self.reset()

        if self.cv_image is not None:
            lines = self.hough.do_hough_line_transform(self.cv_image)
            self.hough.draw_hough_lines(self.cv_image, lines)
            if lines is not None:
                self.calc_lane_lines(lines)
                if self.left and self.right:
                    self.calc_lane_intersection()
                self.visualize_lanes()
        return self.drive(), self.cv_image

    def loop_wrapper(self):
        """ This function takes care of calling the run_loop function repeatedly.
            We are using a separate thread to run the loop_wrapper to work around
            issues with single threaded executors in ROS2 """
        cv2.namedWindow('video_window', 0)
        cv2.resizeWindow('video_window', 800, 500)
        while True:
            self.run_lane_follower()
            self.run_loop()
            time.sleep(0.1)

    def run_loop(self):
        """ Handles showing visualizations and calibration if needed.

            NOTE: only do cv2.imshow and cv2.waitKey in this function 
        """
        if not self.cv_image is None:
            cv2.imshow('video_window', self.cv_image)
            cv2.waitKey(5)

            if self.calibrate_mask:
                self.hough.polygon_pts = []
                cv2.setMouseCallback('video_window', self.update_lane_detection_area)
                cv2.waitKey(0)
                self.calibrate_mask = self.hough.update_lane_mask()  