import cv2 
import numpy as np
import os

class LaneDetection():
    '''
    This program can be used for selecting a color mask to detect traffic signs
    and selecting an epsilon value for polygon classification (to differentiate between signs).
    Signs that we'll include in this project:
    * Yield (Triangle)
    * Traffic Light Ahead (Square)
    * Stop (Hexagon)
    * Do Not Enter (Circle)
    '''
    def __init__(self):
        #(hMin = 124 , sMin = 0, vMin = 169), (hMax = 143 , sMax = 255, vMax = 255)
        # (hMin = 0 , sMin = 44, vMin = 185), (hMax = 179 , sMax = 255, vMax = 255)
        self.directory = "/home/simrun/ros2_ws/images_nov29/left_lane/"
        self.frame_width = 1024
        self.frame_height = 768
        # Set minimum and maximum HSV values to display
        self.lower = np.array([0, 44, 185])
        self.upper = np.array([179, 255, 255])

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


    def find_lane_centers(self):
        for image in os.listdir(self.directory):
            frame = cv2.imread(self.directory + image) 
            # Convert to HSV format and color threshold
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.lower, self.upper)
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # making it gray so we can find contours
            # Applying color mask
            result = cv2.bitwise_and(gray_image, gray_image, mask=mask) 
            cv2.imshow('masked image', result)
            
            # Find contours and detect shape
            contours,_ = cv2.findContours(result, 1, 2)
            
            # find centroids and potentially filter contours by area
            areas = self.find_areas(contours)        
            filtered_contours, centroids = self.filter_contours_find_centroids(contours, areas)
            
            # # draw contours
            # cv2.drawContours(frame, filtered_contours, -1, (0,255,0), 3)
            
            # detect outliers
            true_centroids = self.detect_outliers(centroids)
            # find slope
            slope = self.find_line_fit(true_centroids)
            print(f'Slope: {slope}')
            # draw centroids
            for centroid in true_centroids:
                cv2.circle(frame, (centroid[0], centroid[1]), 7, (0, 0, 255), -1)
            cv2.imshow('frame with contours', frame)  # Display the resulting frame
            cv2.waitKey(0) 

        cv2.destroyAllWindows()


def main(args=None):
    n = LaneDetection()
    n.find_lane_centers()

if __name__ == '__main__':
    main()