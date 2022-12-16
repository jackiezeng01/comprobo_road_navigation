import cv2 
import numpy as np
from geometry_msgs.msg import Twist, Vector3
import time

class ObstacleAvoidance():
    '''
    Class responsbible for the obstacle detection and avoidance behaviour. This
    class in instantiated in neato_car.py
    '''
    def __init__(self):
        self.frame_width = 1024
        self.frame_height = 768
        self.lower = np.array([0, 44, 185]) # minimum HSV values for color mask
        self.upper = np.array([179, 255, 255]) # maximum HSV values for color mask 
        self.cv_image = None
        self.direction = 0
        self.change_lanes_flag = False
        self.centroids = []
        self.filtered_contours = []
        self.first_turn = True
        self.drive_straight = False
        self.second_turn = True
        self.rotation_speed = 0.4
        self.straight_speed = 0.1
        self.start_time = None
        self.twt = Twist()
    
    def filter_contours_find_centroids(self, contours, areas):
        """ 
        Filter contours that are too small and find the centroid of each one
        
        Args:
            contours: list of all the detected contours
            areas: list of corresponding area values for each contour
        """
        for idx,contour in enumerate(contours):        
            # check to see the areas are f
            if areas[idx] > 200:
                M = cv2.moments(contour)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                # check that it is below some point because the lane lines are 
                # in the bottom half of image    
                if cy < 100:           
                    self.centroids.append([cx, cy])
                    self.filtered_contours.append(contour)


    def find_areas(self, contours):
        """
        Find the area of each contour
        
        Args:
            contours: list of all the detected contours 
        
        Returns list of corresponding areas
        """
        areas = []
        for contour in contours:
            area = cv2.contourArea(contour)
            areas.append(area)
        return areas

    def find_line_fit(self, centroids):
        """Find the slope of the line connecting each centroid
        Args:
            centroids: list of points for each centroid
        
        Returns the slope of the line that connects the centroids
        """
        m, _ = np.polyfit(centroids[:,0], -1*centroids[:,1], 1)
        return m

    def detect_obstacles(self, ranges):
        """
        Detecting obstacles based on LIDAR data
        
        Args:
            ranges: list of 360 distances corresponding to the LIDAR data at each angle

        Returns a boolean representing whether an obstacle has been detected
        """
        
        dist_threshold = 0.5 
        point_threshold = 10
        ob_points = 0
        obstacle = False

        # detecting obstacles using above thresholds
        if len(ranges):
            for angle in range(-15,15):
                idx = (angle + 360) % 360 # setting the angles to match NEATO settings
                dist = ranges[idx]
                # considering it to be an obstacle point if it falls below the distance threshold
                if dist < dist_threshold and dist != 0.0:
                    ob_points += 1

                # checking to see how many points were found
                if ob_points > point_threshold:
                    obstacle = True
                    return obstacle
        return obstacle

    def detect_contours(self, image):
        """
        Applies a color mask onto image and detects contours that represent the lane divider lines

        Args:
            image: processed cv image of what the NEATO is seeing
        """
        self.cv_image = image
        if self.cv_image is not None:

            # Convert to HSV format and color threshold
            hsv = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.lower, self.upper)
            gray_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY) # making it gray so we can find contours
            
            # Applying color mask
            result = cv2.bitwise_and(gray_image, gray_image, mask=mask) 
            
            # Find contours
            contours,_ = cv2.findContours(result, 1, 2)
            
            # find centroids and filter contours by area
            areas = self.find_areas(contours)        
            self.filter_contours_find_centroids(contours, areas)
            
            # draw centroids on image
            for centroid in self.centroids:
                cv2.circle(self.cv_image, (centroid[0], centroid[1]), 7, (0, 0, 255), -1)

    def find_slope(self, image):
        """Find the slope based on centroids"""
        self.detect_contours(image)

        # find slope if there at least two centroids detected
        slope = 0
        if len(self.centroids)>= 2:
            slope = self.find_line_fit(np.array(self.centroids))
            print(f'Slope: {slope}')

        return slope

    def find_turn_direction(self, slope):
        """Finding which way to turn based on the slope"""
        
        if slope > 0:
            return 1 # left 
        else:
            return -1 # right


    def change_lanes(self):
        """
        Directs the behaviour for changing lanes which involves turning 90 degrees,
        going straight and then turning 90 degrees in the opposite direction. 
        
        Returns the velocity that the NEATO should turn at as a Twist message
        """

        # turn based on the value set in roation speed for 3 seconds 
        if self.first_turn:            
            if time.time() - self.start_time > 3:
                self.first_turn = False
                self.start_time = None
                self.drive_straight = True
                self.twt.linear = Vector3(x=0.0, y=0.0, z=0.0)
                self.twt.angular = Vector3(x=0.0, y=0.0, z=0.0)
                return self.twt
            else: 
                self.twt.linear = Vector3(x=0.0, y=0.0, z=0.0)
                self.twt.angular = Vector3(x=0.0, y=0.0, z=self.rotation_speed)
                return self.twt
        
        # drive straight for four seconds
        if self.drive_straight:
            if time.time() - self.start_time > 4:
                self.drive_straight = False
                self.second_turn = True
                self.start_time = None
                self.twt.linear = Vector3(x=0.0, y=0.0, z=0.0)
                self.twt.angular = Vector3(x=0.0, y=0.0, z=0.0)
                return self.twt
            else:
                self.twt.linear = Vector3(x=self.straight_speed, y=0.0, z=0.0)
                self.twt.angular = Vector3(x=0.0, y=0.0, z=0.0)
                return self.twt
        
        # turn 90 degrees in the other direction for 3 seconds
        if self.second_turn:
            if time.time() - self.start_time > 3:
                self.second_turn = False
                self.start_time = None
                self.change_lanes_flag = False
                self.twt.linear = Vector3(x=0.0, y=0.0, z=0.0)
                self.twt.angular = Vector3(x=0.0, y=0.0, z=0.0)
                return self.twt
            else: 
                self.twt.linear = Vector3(x=0.0, y=0.0, z=0.0)
                self.twt.angular = Vector3(x=0.0, y=0.0, z=-self.rotation_speed)
                return self.twt


    def obstacle_behaviour(self, ranges, image):
        """ 
        Function that controls the overall obstacle avoidance behaviour. This
        is called in the main loop in neato_car. 
        
        Args:
            ranges: list of 360 distances corresponding to the LIDAR data at each angle
            image: processed cv image of what the NEATO is seeing

        Returns the velocity as a Twist message or None. Also returns the cv image with centroids 
        plotted onto it. 
        """

        # if flag is set to change lanes, then set the start time and 
        # enter the change lanes behavior
        if self.change_lanes_flag:
            if self.start_time is None:
                self.start_time = time.time()
            self.twt = self.change_lanes() 
            # return velocity back to main loop
            return self.twt, self.cv_image
        else:
            # otherwise, detect obstacles
            obstacle_detected = self.detect_obstacles(ranges)
            
            # If an obstacle is detected, find the turning direction and 
            # set the change lanes flag to be true. Return None as the velocity
            # if no obstacles are detected.
            if obstacle_detected:
                print('obstacle detected')
                slope = self.find_slope(image)
                if slope:
                    self.direction = self.find_turn_direction(slope)
                    print("Direction", self.direction)
                    self.rotation_speed = self.direction * self.rotation_speed
                    self.change_lanes_flag = True
                    return None, self.cv_image
                else:
                    return None, self.cv_image
            else:
                return None, self.cv_image
        

    