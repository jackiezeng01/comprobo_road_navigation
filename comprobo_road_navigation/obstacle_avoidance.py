import cv2 
import numpy as np
import os
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist, Vector3, Quaternion
import math

class ObstacleAvoidance():
    '''
    Obstacle Detection
    '''
    def __init__(self, pub):
        self.directory = "/home/simrun/ros2_ws/images_nov29/right_lane/"
        self.frame_width = 1024
        self.frame_height = 768
        # Set minimum and maximum HSV values to display
        self.lower = np.array([0, 44, 185])
        self.upper = np.array([179, 255, 255])
        self.bridge = CvBridge()   
        # self.image_sub = image_sub
        self.vel_pub = pub
        self.cv_image = None
        self.direction = 0
        self.change_lanes_flag = False
        self.start_orientation = None
        self.start_position = None
        self.orientation = None
        self.position = None
        self.first_turn = True
        self.drive_straight = False
        self.second_turn = True
        self.rotation_speed = 0.4
        self.straight_speed = 0.1
        self.twt = Twist()
    
    def process_image(self, msg):
        """ Process image messages from ROS and stash them in an attribute
            called cv_image for subsequent processing """
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def filter_contours_find_centroids(self, contours, areas):
        centroids = []
        filtered_contours = []
        for idx,contour in enumerate(contours):        
            # check the area
            if areas[idx] > 200 and areas[idx] < 4000:
                M = cv2.moments(contour)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                # check that it is below some point because the lane lines are 
                # in the bottom half of image               
                centroids.append([cx, cy])
                filtered_contours.append(contour)
        return filtered_contours, centroids

    def find_areas(self, contours):
        areas = []
        for contour in contours:
            area = cv2.contourArea(contour)
            areas.append(area)
        return areas

    def find_line_fit(self, centroids):
        m, _ = np.polyfit(centroids[:,0], -1*centroids[:,1], 1)
        return m

    def detect_outliers(self, centroids):
        true_centroids = []
        centroids_np = np.array(centroids)
        Q1_x = np.percentile(centroids_np[:,0], 25, interpolation = 'midpoint')
        Q3_x = np.percentile(centroids_np[:,0], 75, interpolation = 'midpoint')
        Q1_y = np.percentile(centroids_np[:,1], 25, interpolation = 'midpoint')
        Q3_y = np.percentile(centroids_np[:,1], 75, interpolation = 'midpoint')
        IQR_x = Q3_x - Q1_x
        IQR_y = Q3_y - Q1_y

        for centroid in centroids:
            if centroid[0] >= (Q3_x + 1.5*IQR_x) or \
            centroid[0] <= (Q1_x - 1.5*IQR_x) or \
            centroid[1] >= (Q3_y + 1.5*IQR_y) or \
            centroid[1] <= (Q1_y - 1.5*IQR_y):
                continue
        else:
                true_centroids.append(centroid)
        return np.array(true_centroids)

    def detect_obstacles(self, ranges):
        dist_threshold = 0.4
        point_threshold = 10
        ob_points = 0
        obstacle = False
        if len(ranges):
            for angle in range(-15,15):
                idx = (angle + 360) % 360
                dist = ranges[idx]
                # print(f'Angle: {angle}, Distance: {dist}')
                if dist < dist_threshold and dist != 0.0:
                    ob_points += 1
                    # print(f'Num points: {ob_points}')
                if ob_points > point_threshold:
                    obstacle = True
                    return obstacle
        return obstacle


    def find_slope(self, image):
        self.cv_image = image
        if self.cv_image is not None:
            print("found image")
            # frame = self.cv_image
            # Convert to HSV format and color threshold
            hsv = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.lower, self.upper)
            gray_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY) # making it gray so we can find contours
            # Applying color mask
            result = cv2.bitwise_and(gray_image, gray_image, mask=mask) 
            # cv2.imshow('masked image', result)
            
            # Find contours and detect shape
            contours,_ = cv2.findContours(result, 1, 2)
            
            # find centroids and potentially filter contours by area
            areas = self.find_areas(contours)        
            filtered_contours, centroids = self.filter_contours_find_centroids(contours, areas)

            arr = np.array(centroids)
            max_idx = np.argmax(arr, axis=0)
            _, max_y = arr[max_idx]
            if max_y[0] < self.frame_width /2:
                slope = 1
            else:
                slope = -1

            # detect outliers
            # true_centroids = self.detect_outliers(centroids)
            # find slope
            slope = 0
            if len(centroids)>= 2:
                slope = self.find_line_fit(np.array(centroids))
                print(f'Slope: {slope}')

            # arr = np.array(centroids)
            # max_idx = np.argmax(arr, axis=0)
            # _, max_y = arr[max_idx]
            # if max_y[0] < self.frame_width /2:
            #     slope = 1
            # else:
            #     slope = -1

            # draw centroids
            for centroid in centroids:
                cv2.circle(self.cv_image, (centroid[0], centroid[1]), 7, (0, 0, 255), -1)

            return slope

    def find_turn_direction(self, slope):
        if slope > 0:
            print("turning left")
            return 1 # left ( direction of angular speed)
        else:
            print("turning right")
            return -1 # right


    def change_lanes(self):
            # set rotation speed 
            if self.first_turn:            
                if abs(self.start_orientation.z - self.orientation.z) >= math.pi/2:
                    self.first_turn = False
                    self.start_orientation = None
                    self.drive_straight = True
                    self.twt.linear = Vector3(x=0.0, y=0.0, z=0.0)
                    self.twt.angular = Vector3(x=0.0, y=0.0, z=0.0)
                    return self.twt
                else: 
                    self.twt.linear = Vector3(x=0.0, y=0.0, z=0.0)
                    self.twt.angular = Vector3(x=0.0, y=0.0, z=self.rotation_speed)
                    return self.twt
            if self.drive_straight:
                if math.dist([self.start_position.x, self.start_position.y], [self.position.x, self.position.y]) > 0.3:
                    self.drive_straight = False
                    self.second_turn = True
                    self.start_orientation = None
                    self.twt.linear = Vector3(x=0.0, y=0.0, z=0.0)
                    self.twt.angular = Vector3(x=0.0, y=0.0, z=0.0)
                    return self.twt
                else:
                    self.twt.linear = Vector3(x=self.straight_speed, y=0.0, z=0.0)
                    self.twt.angular = Vector3(x=0.0, y=0.0, z=0.0)
                    return self.twt
            if self.second_turn:
                if abs(self.start_orientation.z - self.orientation.z) >= math.pi/2:
                    self.second_turn = False
                    self.start_orientation = None
                    self.change_lanes_flag = False
                    self.twt.linear = Vector3(x=0.0, y=0.0, z=0.0)
                    self.twt.angular = Vector3(x=0.0, y=0.0, z=0.0)
                    return self.twt
                else: 
                    self.twt.linear = Vector3(x=0.0, y=0.0, z=0.0)
                    self.twt.angular = Vector3(x=0.0, y=0.0, z=-self.rotation_speed)
                    return self.twt


    def obstacle_behaviour(self, ranges, image, orientation, position):
        "returns velocities, in the form of twist"
        self.orientation = orientation
        self.position = position
        if self.change_lanes_flag:
            if self.start_orientation is None:
                self.start_orientation = self.orientation
                self.start_position = self.position 
            self.twt = self.change_lanes() 
            return self.twt 
        else:
            obstacle_detected = self.detect_obstacles(ranges)
            if obstacle_detected:
                print('obstacle detected')
                slope = self.find_slope(image)
                if slope:
                    self.direction = self.find_turn_direction(slope)
                    print("Direction", self.direction)
                    self.rotation_speed = self.direction * self.rotation_speed
                    self.change_lanes_flag = True
                    return None
                else:
                    return None
            else:
                return None
        

        

# def main(args=None):
#     n = ObstacleAvoidance()
#     n.find_lane_centers()

# if __name__ == '__main__':
#     main()