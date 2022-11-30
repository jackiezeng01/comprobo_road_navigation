import cv2
import numpy as np

class ShapeClassifier():
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
        self.red_lower_bound = 180
        self.red_upper_bound = 255
        self.green_lower_bound = 28
        self.green_upper_bound = 100
        self.blue_lower_bound = 115
        self.blue_upper_bound = 250
        self.shape_epsilon = 2

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

    def run_tracker(self):
        # Define a video capture object and range bars for selecting mask and epsilon value for shape detection
        vid = cv2.VideoCapture(0)
        cv2.namedWindow('mask_selection_window')
        cv2.createTrackbar('red lower bound', 'mask_selection_window', self.red_lower_bound, 255, self.set_red_lower_bound)
        cv2.createTrackbar('red upper bound', 'mask_selection_window', self.red_upper_bound, 255, self.set_red_upper_bound)
        cv2.createTrackbar('green lower bound', 'mask_selection_window', self.green_lower_bound, 255, self.set_green_lower_bound)
        cv2.createTrackbar('green upper bound', 'mask_selection_window', self.green_upper_bound, 255, self.set_green_upper_bound)
        cv2.createTrackbar('blue lower bound', 'mask_selection_window', self.blue_lower_bound, 255, self.set_blue_lower_bound)
        cv2.createTrackbar('blue upper bound', 'mask_selection_window', self.blue_upper_bound, 255, self.set_blue_upper_bound)
        cv2.createTrackbar('shape epsilon', 'mask_selection_window', self.shape_epsilon, 10, self.set_shape_epsilon)

        while(True):
            
            # Capture video frame by frame
            ret, frame = vid.read()
            binary_image = cv2.inRange(frame, (self.blue_lower_bound,self.green_lower_bound,self.red_lower_bound), (self.blue_upper_bound,self.green_upper_bound,self.red_upper_bound))
            cv2.imshow('mask_selection_window', binary_image)

            # Find contours and detect shape
            cnts = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            for c in cnts:
                # Identify shape
                if cv2.contourArea(c) > 35:
                    shape = self.detect_shape(c)
                    # Find centroid and label shape name
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        cv2.putText(frame, shape, (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 2)
            cv2.drawContours(frame, cnts, -1, (0,255,0), 3)

            cv2.imshow('vid_frame', frame)  # Display the resulting frame
            if cv2.waitKey(1) & 0xFF == ord('q'):   # Exit with input 'q'
                break

        # Release the video capture object and destroy all the windows
        vid.release()
        cv2.destroyAllWindows()

    def get_contours(self, binary_image):
        # Find contours and detect shape
        cnts = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        return cnts

def main(args=None):
    n = ShapeClassifier()
    n.run_tracker()

if __name__ == '__main__':
    main()