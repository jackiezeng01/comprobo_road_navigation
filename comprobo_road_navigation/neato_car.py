import rclpy
from threading import Thread
from rclpy.node import Node
import time
import math
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from copy import deepcopy
from cv_bridge import CvBridge
import cv2
import comprobo_road_navigation.helper_functions as helper_functions
import numpy as np
from geometry_msgs.msg import Twist, Vector3, Quaternion
# from comprobo_road_navigation.shape_classification import ShapeClassifier
from comprobo_road_navigation.obstacle_avoidance import ObstacleAvoidance

class NeatoCar(Node):
    """ The BallTracker is a Python object that encompasses a ROS node 
        that can process images from the camera and search for a ball within.
        The node will issue motor commands to move forward while keeping
        the ball in the center of the camera's field of view. """

    def __init__(self, image_topic):
        """ Initialize the ball tracker """
        super().__init__('ball_tracker')
        self.cv_image = None                        # the latest image from the camera
        self.bridge = CvBridge()                    # used to convert ROS messages to OpenCV
        self.sub_image = self.create_subscription(Image, image_topic, self.process_image, 10)
        self.sub_scan = self.create_subscription(LaserScan, 'scan', self.process_laserscan, 10)
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.change_lanes_flag = False
        self.ranges = []
        # rotation stuff
        self.start_orientation = None
        self.rotation_speed = 0.3
        # self.shape_classifier = ShapeClassifier()
        self.obstacle_avoidance = ObstacleAvoidance(self.pub)
        self.turning_flag = 0
        thread = Thread(target=self.loop_wrapper)
        thread.start()

    def get_Odom(self, msg):
        self.position = msg.pose.pose.position
        self.orientation = helper_functions.euler_from_quaternion(msg.pose.pose.orientation)

    def process_image(self, msg):
        """ Process image messages from ROS and stash them in an attribute
            called cv_image for subsequent processing """
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def process_laserscan(self, msg):
        self.ranges = msg.ranges
 

    def loop_wrapper(self):
        """ This function takes care of calling the run_loop function repeatedly.
            We are using a separate thread to run the loop_wrapper to work around
            issues with single threaded executors in ROS2 """
        # cv2.namedWindow('frame_with_centroids', 0)
        # cv2.resizeWindow('frame_with_centroids', 800, 500)
        while True:
            self.run_loop()
            time.sleep(1)
        # cv2.destroyAllWindows()

    def turn_ninety_deg(self):
        # set rotation speed 
        if abs(self.start_orientation.z - self.orientation.z) >= math.pi/2:
            self.turning_flag = 0
            self.start_orientation = None
            return Vector3(x=0.0, y=0.0, z=0.0)
        else: 
            return Vector3(x=0.0, y=0.0, z=self.rotation_speed)

    def run_loop(self):
        # NOTE: only do cv2.imshow and cv2.waitKey in this function 
        velocity = Twist()
        velocity.linear.x = float(0.1)
        # if the turning flag is set to 1, the robot will turn until it reaches 90 degrees
        if self.turning_flag:
            if self.start_orientation is None:
                self.start_orientation = self.orientation
            velocity.linear = Vector3(x=0.0, y=0.0, z=0.0)
            velocity.angular = self.turn_ninety_deg()
        if not self.change_lanes_flag:
            self.turn_speed = 0
            self.turn_speed, self.change_lanes_flag = self.obstacle_avoidance.obstacle_behaviour(self.ranges, self.cv_image, self.change_lanes_flag)
        if self.change_lanes_flag:
            velocity.angular.z = float(self.turn_speed)


        print(f'Linear speed{velocity.linear.x}, Angular speed: {velocity.angular.z}')
        self.pub.publish(velocity)
            

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