import rclpy
from threading import Thread
from rclpy.node import Node
import time
from sensor_msgs.msg import Image
import apriltag
from comprobo_road_navigation.path_planning import PathPlanning
from cv_bridge import CvBridge
import cv2
from enum import Enum
from geometry_msgs.msg import Twist

Neato_state = Enum('Neato_state', ['FOLLOWING_INSTRUCTION', 'TURNING'])

class AprilTagDetector(Node):
    """ The AprilTagDetector is a Python object that encompasses a ROS node 
        that can process images from the camera and search for apriltags.
        The node will issue motor commands
        1. Move forward until it reaches the target apriltag
        2. Follow the path planned instructions to turn left or right once
        it has reached the target apriltag """

    def __init__(self, image_topic):
        """ Initialize the apriltag detector """
        super().__init__('apriltag_detector')
        # Set up path planning, get instructions
        self.apriltag_stop_distances = {0: 2600,
                                        1: 2600,
                                        2: 2300,
                                        3: 2600,
                                        4: 2300,
                                        5: 2600,
                                        6: 2300,
                                        7: 2600,
                                        8: 2600}
        tag_map = {((1, 0), (2, 0)): 6, ((2, 0), (3, 0)): 6, ((2, 0), (2, 1)): 5,
                    ((0, 1), (0, 2)): 2, ((0, 2), (0, 3)): 2, ((1, 2), (2, 2)): 3,
                    ((2, 1), (2, 2)): 4, ((2, 2), (2, 3)): 4, ((0, 4), (0, 5)): 1,
                    ((2, 5), (3, 5)): 8}
        self.pathplanner = PathPlanning(tag_map)
        start_node = (3, 5)
        end_node = (0, 2)
        self.pathplanner.node_to_node(start_node, end_node)
        print("path: ", self.pathplanner.path)
        self.pathplanner.generate_instructions()
        print("instructions: ", self.pathplanner.instructions)
        self.state = Neato_state.FOLLOWING_INSTRUCTION

        # Set up ROS image input, subscribers and publishers
        self.cv_image = None                        # the latest image from the camera
        self.bridge = CvBridge()                    # used to convert ROS messages to OpenCV
        self.create_subscription(Image, image_topic, self.process_image, 10)
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)
        thread = Thread(target=self.loop_wrapper)
        thread.start()

    def process_image(self, msg):
        """ Process image messages from ROS and stash them in an attribute
            called cv_image for subsequent processing """
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def detect_apriltags(self):
        """
        Detects apriltags in the camera view and returns them as a
        dictionary with tag IDs as keys and sizes as values
        """
        gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
        options = apriltag.DetectorOptions(families="tag36h11")
        detector = apriltag.Detector(options)
        results = detector.detect(gray)
        apriltags = {}

        # loop over the AprilTag detection results
        for r in results:
            # extract the bounding box (x, y)-coordinates for the AprilTag
            # and convert each of the (x, y)-coordinate pairs to integers
            (ptA, ptB, ptC, ptD) = r.corners
            ptB = (int(ptB[0]), int(ptB[1]))
            ptC = (int(ptC[0]), int(ptC[1]))
            ptD = (int(ptD[0]), int(ptD[1]))
            ptA = (int(ptA[0]), int(ptA[1]))
            # draw the bounding box of the AprilTag detection
            cv2.line(self.cv_image, ptA, ptB, (0, 255, 0), 2)
            cv2.line(self.cv_image, ptB, ptC, (0, 255, 0), 2)
            cv2.line(self.cv_image, ptC, ptD, (0, 255, 0), 2)
            cv2.line(self.cv_image, ptD, ptA, (0, 255, 0), 2)
            # draw the center (x, y)-coordinates of the AprilTag
            (cX, cY) = (int(r.center[0]), int(r.center[1]))
            cv2.circle(self.cv_image, (cX, cY), 5, (0, 0, 255), -1)
            # draw the tag family on the image
            tagFamily = r.tag_family.decode("utf-8")
            tagID = r.tag_id
            size = pow(int(ptB[0]) - int(ptA[0]), 2)
            cv2.putText(self.cv_image, str(tagID), (ptA[0], ptA[1] + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 50), 2)
            cv2.putText(self.cv_image, str(size), (ptB[0], ptB[1] - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            apriltags[tagID] = size
        return apriltags

    def loop_wrapper(self):
        """ This function takes care of calling the run_loop function repeatedly.
            We are using a separate thread to run the loop_wrapper to work around
            issues with single threaded executors in ROS2 """
        cv2.namedWindow('video_window', 0)
        cv2.resizeWindow('video_window', 800, 500)
        while True and self.cv_image is not None:
            """
            Right now, the behavior is to keep going straight at 0.1 speed until the target apriltag
            is bigger than the size listed in self.apriltag_stop_distances (i.e. the apriltag is the
            target distance away)

            What we need to do next is combine this with line following + path planning instructions.
            The end behavior should be:
            while path planning instructions != []:
                Pop first item from path planning instructions (e.g. Turn left at april id 5)
                target apriltag = 5
                target distance = self.apriltag_stop_distances.get(target apriltag)
                While (the target apriltag is not the target distance away):
                    keep doing line following
                    if (target apriltag is the target distance away):
                        turn behavior (left, based on instructions)
                        break
            """
            msg = Twist()
            # print(aprilTags)
            if self.state == Neato_state.TURNING:
                current_instruction = self.instructions[0]
                # current_instruction = self.instructions[0]
                print("current_instruction: ", current_instruction)
                aprilTag_to_look_for = current_instruction[0]
                action_at_aprilTag = current_instruction[1]
                size_at_which_to_stop = self.apriltag_stop_distances.get(aprilTag_to_look_for)
                aprilTags = self.detect_apriltags()
                print("FOLLOWING_INSTRUCTION")
                msg.linear.x = 0.1
                if aprilTag_to_look_for in aprilTags.keys():
                    print("size aprilTag_to_look_for", aprilTags.get(aprilTag_to_look_for))
                    # if we have seen the apriltag we are looking for at the right distance away
                    if aprilTags.get(aprilTag_to_look_for) >= size_at_which_to_stop:
                        msg.linear.x = 0.0
                        self.instructions.pop(0)
                        self.state = Neato_state.TURNING
            if self.state == Neato_state.TURNING:
                action_at_aprilTag == 'right'
                print("TURNING")
                t = time.time()
                if (time.time() - t < 5):
                    msg.linear.x = 0.1
                    if action_at_aprilTag == 'right':
                        msg.angular.z = -0.05
                    if action_at_aprilTag == 'left':
                        msg.angular.z = 0.05
                else:
                    self.state == Neato_state.FOLLOWING_INSTRUCTION
                    msg.linear.x = 0.0
                    msg.angular.z = 0.0
            self.pub.publish(msg)
            self.run_loop()
            time.sleep(0.1)

    def run_loop(self):
        # NOTE: only do cv2.imshow and cv2.waitKey in this function 
        if not self.cv_image is None:
            cv2.imshow('video_window', self.cv_image)
            # if hasattr(self, 'image_info_window'):
            #     cv2.imshow('image_info', self.image_info_window)
            cv2.waitKey(5)

if __name__ == '__main__':
    node = AprilTagDetector("/camera/image_raw")
    node.run()

def main(args=None):
    rclpy.init()
    n = AprilTagDetector("camera/image_raw")
    rclpy.spin(n)
    rclpy.shutdown()


if __name__ == '__main__':
    main()