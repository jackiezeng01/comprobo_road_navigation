import rclpy
from threading import Thread
from rclpy.node import Node
import time
import math
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Vector3
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2
import comprobo_road_navigation.helper_functions as helper_functions
from comprobo_road_navigation.obstacle_avoidance import ObstacleAvoidance
from comprobo_road_navigation.path_planning import PathPlanning
from comprobo_road_navigation.roadsign_detector import RoadSignDetector
from comprobo_road_navigation.apriltag_detector import AprilTagDetector
from comprobo_road_navigation.helper_functions import undistort_img
from comprobo_road_navigation.lane_following import Lane_Follower

class NeatoCar(Node):
    """ The NeatoCar is a Python object that encompasses a ROS node 
        that can process images from the camera and autonomously navigate an indoor track, 
        while following road traffic rules.
    """

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
        self.image_obstacles = None
        self.last_instruction = None
        self.rotation_speed = 0.3
        self.linear_speed = 0.1
        start_node = (0, 5)
        end_node = (2, 2)
        self.pathplanner = PathPlanning(start_node, end_node)
        self.pathplanner.node_to_node((2, 2), (5, 4))
        self.instructions = self.pathplanner.generate_instructions()
        self.obstacle_avoidance = ObstacleAvoidance(self.pub)
        self.apriltag_detector = AprilTagDetector()
        self.roadsign_detector = RoadSignDetector()
        self.lane_follower = Lane_Follower()
        self.velocity = None
        self.turning_flag = False
        self.drive_straight = True
        self.turn = False
        self.in_double_lane = False
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
        while True:
            # print("looping")
            if self.orientation and self.position and self.cv_image is not None:
                if self.last_instruction == 0 or \
                   self.last_instruction == 1 or \
                   self.last_instruction == 7 or \
                   self.last_instruction == 9:
                    print("entering double lane")
                    self.velocity = self.obstacle_avoidance.obstacle_behaviour(self.ranges, self.cv_image, self.orientation, self.position)
                if self.turning_flag is False:
                    if self.instructions != []:
                        instruction = self.instructions[0]
                        reached, self.cv_image = self.apriltag_detector.run_apriltag_detector(self.cv_image, self.raw_cv_image, instruction)
                        print("reached: ", reached)
                        # If we have reached an apriltag which has a turn instruction
                        if reached == 1:
                            self.turning_flag = True
                        # If we have reached an apriltag which has a go straight instruction
                        if reached == 2:
                            self.last_instruction = self.instructions[0][0]
                            self.instructions.pop(0)
                    self.velocity, self.cv_image = self.lane_follower.run_lane_follower(self.cv_image, self.linear_speed, self.rotation_speed, self.orientation, self.position)
                if self.turning_flag is True:
                    self.turning_behaviour(instruction[1])
                if self.velocity is not None:
                    self.pub.publish(self.velocity)
                    
            self.run_loop()
            time.sleep(0.1)
        # cv2.destroyAllWindows()

    def get_roadsign_action(self):
        """ return 'go' or 'stop' based on the results from roadsign detection.
        """
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
            if time.time() - self.start_time > 5:
                self.drive_straight = False
                self.turn = True
                self.start_time = None
                self.velocity.linear = Vector3(x=0.0, y=0.0, z=0.0)
                self.velocity.angular = Vector3(x=0.0, y=0.0, z=0.0)
                return self.velocity
            else:
                self.velocity.linear = Vector3(x=self.linear_speed, y=0.0, z=0.0)
                self.velocity.angular = Vector3(x=0.0, y=0.0, z=0.0)
                return self.velocity
        if self.turn:
            if time.time() - self.start_time >= 5:
                self.drive_straight = True
                self.turn = False
                self.start_time = None
                self.turning_flag = False
                self.velocity.linear = Vector3(x=0.0, y=0.0, z=0.0)
                self.velocity.angular = Vector3(x=0.0, y=0.0, z=0.0)
                self.last_instruction = self.instructions[0][0]
                self.instructions.pop(0)
                return self.velocity 
            else:  
                self.velocity.linear = Vector3(x=0.0, y=0.0, z=0.0)
                if direction == "right":
                    self.velocity.angular = Vector3(x=0.0, y=0.0, z= -abs(self.rotation_speed))
                if direction == "left":
                    self.velocity.angular = Vector3(x=0.0, y=0.0, z= abs(self.rotation_speed))
                return self.velocity

    def turning_behaviour(self, direction):
        if self.turning_flag:
            self.velocity = Twist()
            if self.start_time is None:
                print("resetting time")
                self.start_time = time.time()
            self.velocity = self.turning(direction)

    def run_loop(self):
        if self.cv_image is not None or self.image_obstacles is not None:
            if self.cv_image is not None:
                cv2.imshow('video_window', self.cv_image)
            if self.image_obstacles is not None:
                cv2.imshow('obst window', self.image_obstacles)
            cv2.waitKey(5)
    
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