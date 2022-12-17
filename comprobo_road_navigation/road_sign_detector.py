import rclpy
from threading import Thread
from rclpy.node import Node
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from geometry_msgs.msg import Twist

class RoadSignDetector(Node):
    """ The BallTracker is a Python object that encompasses a ROS node 
        that can process images from the camera and search for a ball within.
        The node will issue motor commands to move forward while keeping
        the ball in the center of the camera's field of view. """

    def __init__(self, image_topic):
        """ Initialize the ball tracker """
        super().__init__('ball_tracker')
        self.cv_image = None                        # the latest image from the camera
        self.bridge = CvBridge()                    # used to convert ROS messages to OpenCV
        self.road_sign_id_confidence = 0
        self.signs_detected = []
        self.cX = 0
        self.cY = 0

        self.create_subscription(Image, image_topic, self.process_image, 10)
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)
        # """
        # road sign bounds
        # self.red_lower_bound = 0
        # self.red_upper_bound = 160
        # self.green_lower_bound = 127
        # self.green_upper_bound = 225
        # self.blue_lower_bound = 0
        # self.blue_upper_bound = 80
        # """

        # traffic light green bounds
        self.red_lower_bound = 0
        self.red_upper_bound = 8
        self.green_lower_bound = 42
        self.green_upper_bound = 92
        self.blue_lower_bound = 0
        self.blue_upper_bound = 46

        self.shape_epsilon = 4
        thread = Thread(target=self.loop_wrapper)
        thread.start()

    def set_red_lower_bound(self, val):
        """ A callback function to handle the OpenCV slider to select the red lower bound """
        self.red_lower_bound = val

    def set_red_upper_bound(self, val):
        """ A callback function to handle the OpenCV slider to select the red upper bound """
        self.red_upper_bound = val

    def set_green_lower_bound(self, val):
        """ A callback function to handle the OpenCV slider to select the green lower bound """
        self.green_lower_bound = val

    def set_green_upper_bound(self, val):
        """ A callback function to handle the OpenCV slider to select the green upper bound """
        self.green_upper_bound = val

    def set_blue_lower_bound(self, val):
        """ A callback function to handle the OpenCV slider to select the blue lower bound """
        self.blue_lower_bound = val

    def set_blue_upper_bound(self, val):
        """ A callback function to handle the OpenCV slider to select the blue upper bound """
        self.blue_upper_bound = val

    def set_shape_epsilon(self, val):
        """ A callback function to handle the OpenCV slider to select the shape detection epsilon"""
        self.shape_epsilon = val

    def process_image(self, msg):
        """ Process image messages from ROS and stash them in an attribute
            called cv_image for subsequent processing """
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def detect_shape(self, c):
        # Compute perimeter of contour and perform contour approximation
        sign = ""
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, self.shape_epsilon/100 * peri, True)
        num_edges = len(approx)
        if num_edges == 3: # Triangle
            sign = f"Yield {num_edges}"
        elif num_edges == 4: # Square or rectangle
            sign = f"Traffic Light Ahead {num_edges}"
        elif num_edges >= 5 and num_edges <= 8:
            sign = f"Stop {num_edges}"
        # Otherwise assume as circle or oval
        else:
            sign = f"Do Not Enter {num_edges}"
        return sign
        
    def get_contours(self, binary_image):
        # Find contours and detect shape
        cnts = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        return cnts

    def id_and_draw_shapes(self, frame, cnts):
        for c in cnts:
            # Identify shape
            if cv2.contourArea(c) > 150:
                # shape = self.detect_shape(c)
                # Find centroid and label shape name
                M = cv2.moments(c)
                if M["m00"] != 0:
                    self.cX = int(M["m10"] / M["m00"])
                    self.cY = int(M["m01"] / M["m00"])
                    # cv2.putText(frame, shape, (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (36,4,12), 2)
        cv2.drawContours(frame, cnts, -1, (0,255,0), 3)
        # self.signs_detected.append(shape)

    def id_sign_with_confidence(self):
        """
        Add detected roadsign to list of signs detected
        Use this list to see which road sign it probably is (whichever has the highest count)
        and confidence (count of the road sign with the highest count / total count)
        """
        roadsign = max(self.signs_detected, key=self.signs_detected.count)
        self.confidence = self.signs_detected.count(roadsign)/len(self.signs_detected)
        return roadsign

    def loop_wrapper(self):
        """ This function takes care of calling the run_loop function repeatedly.
            We are using a separate thread to run the loop_wrapper to work around
            issues with single threaded executors in ROS2 """
        cv2.namedWindow('video_window', 0)
        cv2.namedWindow('binary_window', 0)
        cv2.resizeWindow('video_window', 800, 500)
        cv2.resizeWindow('binary_window', 800, 500)
        cv2.createTrackbar('red lower bound', 'binary_window', self.red_lower_bound, 255, self.set_red_lower_bound)
        cv2.createTrackbar('red upper bound', 'binary_window', self.red_upper_bound, 255, self.set_red_upper_bound)
        cv2.createTrackbar('green lower bound', 'binary_window', self.green_lower_bound, 255, self.set_green_lower_bound)
        cv2.createTrackbar('green upper bound', 'binary_window', self.green_upper_bound, 255, self.set_green_upper_bound)
        cv2.createTrackbar('blue lower bound', 'binary_window', self.blue_lower_bound, 255, self.set_blue_lower_bound)
        cv2.createTrackbar('blue upper bound', 'binary_window', self.blue_upper_bound, 255, self.set_blue_upper_bound)
        cv2.createTrackbar('shape epsilon', 'binary_window', self.shape_epsilon, 10, self.set_shape_epsilon)
        while True and self.cv_image is not None:
            self.binary_image = cv2.inRange(self.cv_image, (self.blue_lower_bound,self.green_lower_bound,self.red_lower_bound), (self.blue_upper_bound,self.green_upper_bound,self.red_upper_bound))
            contours = self.get_contours(self.binary_image)
            self.id_and_draw_shapes(self.cv_image, contours)
            # sign = self.id_sign_with_confidence()
            # if self.confidence > 0.5:
                # cv2.putText(self.cv_image, sign, (self.cX - 20, self.cY), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (36,4,12), 2)
            self.run_loop()
            time.sleep(0.1)

    def run_loop(self):
        # NOTE: only do cv2.imshow and cv2.waitKey in this function 
        if not self.cv_image is None:
            cv2.imshow('video_window', self.cv_image)
            cv2.imshow('binary_window', self.binary_image)
            # if hasattr(self, 'image_info_window'):
            #     cv2.imshow('image_info', self.image_info_window)
            cv2.waitKey(5)

if __name__ == '__main__':
    node = RoadSignDetector("/camera/image_raw")
    node.run()

def main(args=None):
    rclpy.init()
    n = RoadSignDetector("camera/image_raw")
    rclpy.spin(n)
    rclpy.shutdown()


if __name__ == '__main__':
    main()