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
        if areas[idx] > 500 and areas[idx] < 4000:
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
    m, b = np.polyfit(centroids[:,0], -1*centroids[:,1], 1)
    return m,b

def detect_outliers(centroids):
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




    





directory = "/home/simrun/ros2_ws/images_nov29/left_lane/"
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
        # y_cutoff = find_crop_region(contours, areas)

        # alternatively just cut in half and use bottom half
        y_cutoff = int(frame_height/3)

        # create mask
        mask = np.zeros(frame.shape[:2], dtype="uint8")
        cv.rectangle(mask,(0,y_cutoff),(frame_width,frame_height), 255, -1)
        masked = cv.bitwise_and(frame, frame, mask=mask)
        
        # find centroids
        filtered_contours, centroids = filter_contours_find_centroids(contours, areas, y_cutoff)

        # detect outliers
        # centroids = detect_outliers(unfiltered_centroids) # returns centroids as numpy array
        # find left or right
        # if bool(centroids.ndim > 1):
        #     m,b = find_line_fit(centroids)
        #     print(f'Slope: {m}')
        # else:
        #     print("Not enough points to find slope")
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