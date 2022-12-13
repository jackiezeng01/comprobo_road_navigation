import rclpy
from threading import Thread
from rclpy.node import Node
import time
import math
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from copy import deepcopy
from cv_bridge import CvBridge
import cv2
import comprobo_road_navigation.helper_functions as helper_functions
import numpy as np
from geometry_msgs.msg import Twist, Vector3, Quaternion
# from comprobo_road_navigation.shape_classification import ShapeClassifier
from comprobo_road_navigation.obstacle_avoidance import ObstacleAvoidance
from comprobo_road_navigation.path_planning import PathPlanning
from comprobo_road_navigation.roadsign_detector import RoadSignDetector
from comprobo_road_navigation.apriltag_detector import AprilTagDetector
from comprobo_road_navigation.helper_functions import Point, Line, HoughLineDetection, euler_from_quaternion, undistort_img
from comprobo_road_navigation.lane_following import Lane_Detector

class NeatoCar(Node):
    """ The BallTracker is a Python object that encompasses a ROS node 
        that can process images from the camera and search for a ball within.
        The node will issue motor commands to move forward while keeping
        the ball in the center of the camera's field of view. """

    def __init__(self, image_topic):
        """ Initialize the ball tracker """
        super().__init__('ball_tracker')
        self.raw_cv_image = None
        self.cv_image = None                        # the latest image from the camera
        self.bridge = CvBridge()                    # used to convert ROS messages to OpenCV
        self.sub_image = self.create_subscription(Image, image_topic, self.process_image, 10)
        self.sub_scan = self.create_subscription(LaserScan, 'scan', self.process_laserscan, 10)
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.process_odom, 10)
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.change_lanes_flag = False
        self.first_turn = True
        self.second_turn = False
        self.drive_straight = False
        self.ranges = []
        # rotation stuff
        self.start_orientation = None
        self.start_time = None
        self.orientation = None
        self.position = None
        self.rotation_speed = 0.3
        self.linear_speed = 0.1
        start_node = (3, 5)
        end_node = (4, 0)
        self.pathplanner = PathPlanning(start_node, end_node)
        self.instructions = self.pathplanner.generate_instructions()
        print("instructions: ", self.instructions)
        self.obstacle_avoidance = ObstacleAvoidance(self.pub)
        self.apriltag_detector = AprilTagDetector()
        self.roadsign_detector = RoadSignDetector()
        self.lane_detector = Lane_Detector()
        self.velocity = None
        self.turning_flag = False
        self.stopping_flag = False
        self.drive_straight = True
        self.turn = False
        # self.velocity = Twist()
        thread = Thread(target=self.loop_wrapper)
        thread.start()

    def process_odom(self, msg):
        self.position = msg.pose.pose.position
        self.orientation = helper_functions.euler_from_quaternion(msg.pose.pose.orientation)

    def process_image(self, msg):
        """ Process image messages from ROS and stash them in an attribute
            called cv_image for subsequent processing """
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.cv_image = undistort_img(cv_image)
        self.raw_cv_image = undistort_img(cv_image)


    def process_laserscan(self, msg):
        self.ranges = msg.ranges
 
    def loop_wrapper(self):
        """ This function takes care of calling the run_loop function repeatedly.
            We are using a separate thread to run the loop_wrapper to work around
            issues with single threaded executors in ROS2 """
        while True and self.instructions != []:
            # print("looping")
            if self.orientation and self.position:
                self.velocity = self.obstacle_avoidance.obstacle_behaviour(self.ranges, self.cv_image, self.orientation, self.position)
                if self.velocity is None and self.turning_flag is False and self.turning_flag is False:
                    # self.velocity, self.cv_image = self.lane_detector.run_lane_detector(self.cv_image, self.linear_speed, self.rotation_speed, self.orientation, self.position)
                    self.velocity = Twist()
                    self.velocity.linear = Vector3(x=0.1, y=0.0, z=0.0)
                    self.velocity.angular = Vector3(x=0.0, y=0.0, z=0.0)
                    instruction = self.instructions[0]
                    # print("instruction:", instruction)
                    reached, self.cv_image = self.apriltag_detector.run_apriltag_detector(self.cv_image, self.raw_cv_image, instruction)
                    self.roadsign_to_obey = self.roadsign_detector.run_roadsign_detector(self.cv_image, self.raw_cv_image)
                    print("reached: ", reached)
                    # If we have reached an apriltag which has a turn instruction
                    if reached == 1:
                        self.turning_flag = True
                    # If we have reached an apriltag which has a go straight instruction
                    if reached == 2:
                        self.instructions.pop(0)

                    """
                    Integrating roadsign detection
                    
                    # If we have reached an apriltag which has a turn instruction
                    if reached == 1:
                        print("here")
                        action = self.get_roadsign_action()
                        if (action == 'stop'):
                            self.stopping_flag = True
                        else:
                            self.turning_flag = True
                    # If we have reached an apriltag which has a go straight instruction
                    if reached == 2:
                        action = self.get_roadsign_action()
                        if (action == 'stop'):
                            self.stopping_flag = True
                    """
                if self.turning_flag is True:
                    self.turning_behaviour(instruction[1])
                if self.stopping_flag is True:
                    self.stopping_behaviour()
                if self.velocity is not None:
                    self.pub.publish(self.velocity)
                    
            self.run_loop()
            time.sleep(0.1)
        # cv2.destroyAllWindows()

    def get_roadsign_action(self):
        if self.roadsign_to_obey == []:
            return 'go'
        elif 'Yield' in self.roadsign_to_obey or 'Stop' in self.roadsign_to_obey:
            return 'stop'
        return self.roadsign_detector.get_traffic_light_action(self.raw_cv_image)
            

    def turn_ninety_deg(self):
        # set rotation speed 
        if abs(self.start_orientation.z - self.orientation.z) >= math.pi/2:
            self.turning_flag = False
            self.start_orientation = None
            print('inside if')
            return Vector3(x=0.0, y=0.0, z=0.0)
        else: 
            return Vector3(x=0.0, y=0.0, z=self.rotation_speed)
    
    def turning(self, direction):
        if self.drive_straight:
            if time.time() - self.start_time > 7:
                print("her1")
                self.drive_straight = False
                self.turn = True
                self.start_time = None
                self.velocity.linear = Vector3(x=0.0, y=0.0, z=0.0)
                self.velocity.angular = Vector3(x=0.0, y=0.0, z=0.0)
                return self.velocity
            else:
                print("her2")
                self.velocity.linear = Vector3(x=self.linear_speed, y=0.0, z=0.0)
                self.velocity.angular = Vector3(x=0.0, y=0.0, z=0.0)
                return self.velocity
        if self.turn:
            if time.time() - self.start_time >= 5:
                print("her3")
                self.drive_straight = True
                self.turn = False
                self.start_time = None
                self.turning_flag = False
                self.velocity.linear = Vector3(x=0.0, y=0.0, z=0.0)
                self.velocity.angular = Vector3(x=0.0, y=0.0, z=0.0)
                self.instructions.pop(0)
                return self.velocity 
            else:  
                print("her4")               
                self.velocity.linear = Vector3(x=0.0, y=0.0, z=0.0)
                if direction == "right":
                    self.velocity.angular = Vector3(x=0.0, y=0.0, z= -abs(self.rotation_speed))
                if direction == "left":
                    self.velocity.angular = Vector3(x=0.0, y=0.0, z= abs(self.rotation_speed))
                return self.velocity

    def stopping(self):
        if time.time() - self.start_time < 5:
            print("stop function here 1")
            self.velocity.linear = Vector3(x=0.0, y=0.0, z=0.0)
            self.velocity.angular = Vector3(x=0.0, y=0.0, z=0.0)
        else:
            print("stop function here 2")
            self.start_time = None
            self.stopping_flag = False


    def turning_behaviour(self, direction):
        if self.turning_flag: # TODO I think this if else statement is redundant cuz we only call this function is turning flag is true
            self.velocity = Twist()
            if self.start_time is None:
                print("resetting time")
                self.start_time = time.time()
                # self.start_orientation = self.orientation
                # self.start_position = self.position 
            self.velocity = self.turning(direction)
            """
            TODO ^^^ do we need to have self.turning return a velocity?
            it's all in the same class, so we could possibly just modify self.velocity from within the self.turning function
            """
    
    def stopping_behaviour(self):
        # TODO test that this works
        self.velocity = Twist()
        if self.start_time is None:
            print("resetting time")
            self.start_time = time.time()

    def run_loop(self):
        # NOTE: only do cv2.imshow and cv2.waitKey in this function 
        
        # self.velocity.linear.x = self.linear_speed
        # if the turning flag is set to 1, the robot will turn until it reaches 90 degrees
        # if self.turning_flag:
        #     if self.start_orientation is None:
        #         self.start_orientation = self.orientation
        #     self.velocity.linear = Vector3(x=0.0, y=0.0, z=0.0)
        #     self.velocity.angular = self.turn_ninety_deg()
        #     print("vel",self.velocity)
        if not self.cv_image is None:
            cv2.imshow('video_window', self.cv_image)
            cv2.waitKey(5)
            # print(self.calibrate_mask)

        # # self.obstacle_avoidance.find_lane_centers(self.cv_image)
        # if self.obstacle_avoidance.cv_image is not None:
        #     cv2.imshow('frame_with_centroids', self.obstacle_avoidance.cv_image)
        #     cv2.waitKey(5)

if __name__ == '__main__':
    node = NeatoCar("/camera/image_raw")
    node.run()

def main(args=None):
    rclpy.init()
    n = NeatoCar("camera/image_raw")
    rclpy.spin(n)
    rclpy.shutdown()


if __name__ == '__main__':
    main()