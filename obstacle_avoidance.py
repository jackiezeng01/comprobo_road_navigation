import cv2 as cv
import numpy as np
import os


def find_contours(frame):
        # convert image to grayscale image
    gray_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # convert the grayscale image to binary image
    ret,thresh = cv.threshold(gray_image,127,255,0)
    
    # calculate moments of binary image
    M = cv.moments(thresh)
    contours,hierarchy = cv.findContours(thresh, 1, 2)

def find_crop_region(contours, areas):
    max_value = max(areas)
    largest_contour = contours[areas.index(max_value)]
    _,y,_,h = cv.boundingRect(largest_contour)
    return y+h

def filter_contours_find_centroids(contours, areas, y_cutoff):
    centroids = []
    filtered_contours = []
    for idx,contour in enumerate(contours):        
        # check the area
        if areas[idx] > 100 and areas[idx] < 2000:
            M = cv.moments(contour)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            # check that it is below some point because the lane lines are 
            # in the bottom half of image
            if cy > y_cutoff:                
                centroids.append([cx, cy])
                filtered_contours.append(contour)
    return filtered_contours, centroids

def find_line_fit(centroids):
    centroids_array = np.array(centroids)
    m, b = np.polyfit(centroids_array[:,0], -1*centroids_array[:,1], 1)
    return m,b
    





directory = "/home/simrun/ros2_ws/src/comprobo_road_navigation/sample_images/right/"
frame_width = 1024
frame_height = 768

def main():

    for image in os.listdir(directory):
        # Using cv2.imread() method
        frame = cv.imread(directory + image)   
        # convert image to grayscale image
        gray_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # add guausian blur
        blur = cv.GaussianBlur(gray_image, (5, 5), 0)        
        # convert the grayscale image to binary image
        ret,thresh = cv.threshold(gray_image,127,255,0)
        # find contours
        contours,_ = cv.findContours(thresh, 1, 2)
        areas = []
        for contour in contours:
            area = cv.contourArea(contour)
            areas.append(area)

        # use wall as a way to mask image. finds the bounding box off 
        # wall contour and uses that to mask image
        y_cutoff = find_crop_region(contours, areas)

        # alternatively just cut in half and use bottom half
        # y_cutoff = int(frame_height/2)

        # create mask
        mask = np.zeros(frame.shape[:2], dtype="uint8")
        cv.rectangle(mask,(0,y_cutoff),(frame_width,frame_height), 255, -1)
        masked = cv.bitwise_and(frame, frame, mask=mask)
        
        # find centroids
        filtered_contours, centroids = filter_contours_find_centroids(contours, areas, y_cutoff)

        # find left or right
        m,b = find_line_fit(centroids)
        print(f'Slope: {m}')

        # draw centroids
        for centroid in centroids:
            cv.circle(masked, (centroid[0], centroid[1]), 7, (0, 0, 255), -1)
        # draw contours
        cv.drawContours(masked, filtered_contours, -1, (0, 255, 0), 1)
        cv.imshow("contours", masked)
        cv.waitKey(0)
        # create a custom mask for left and right and use that
    cv.destroyAllWindows()


if __name__ == "__main__":
   main()