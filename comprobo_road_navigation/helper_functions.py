import cv2
import numpy as np
from geometry_msgs.msg import Twist, Vector3, Quaternion
import math


def euler_from_quaternion(quaternion):
    """
    From https://automaticaddison.com/how-to-convert-a-quaternion-into-euler-angles-in-python/ 
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x = quaternion.x
    y = quaternion.y
    z = quaternion.z
    w = quaternion.w
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return Vector3(x=roll_x, y=pitch_y, z=yaw_z)  # in radians


def undistort_img(img):
    """ Undistort image using camera calibration parameters.

        Inputs:
            img: raw image 

        Returns:
            dst: undistored image
    """
    method = "basic"
    # image dimensions: (width, height)
    dim = (1024, 768)
    # camera matrix
    mtx = np.array([[511.924979, 0.000000,   498.854696],
                    [0.000000,   512.669071, 346.824822],
                    [0.000000,   0.000000,   1.000000]])
    # distortion coefficients
    dist = np.array([-0.237095, 0.050504, -0.009065, 0.000321, 0.000000])
    # get refined camera matrix
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, dim, 1, dim)
    if (method == "basic"):
        # undistort with cv function
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    else:
        # undistort with remapping
        mapx, mapy = cv2.initUndistortRectifyMap(
            mtx, dist, None, newcameramtx, dim, 5)
        dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst


"""
Lane following helper functions below ------------------------------------------
"""


class Point:
    def __init__(self, x, y) -> None:
        self.x = int(x)
        self.y = int(y)
        self.xy = (int(x), int(y))

    def __str__(self):
        return f"({self.x},{self.y})"


class Line:
    def __init__(self, x1: int, y1: int, x2: int, y2: int) -> None:
        self.pt1 = Point(x1, y1)
        self.pt2 = Point(x2, y2)
        self.arr = [[x1, y1], [x2, y2]]
        self.slope, self.y_intercept = self.fit_linear_polynomial()

    def __str__(self):
        return f"({self.pt1},{self.pt2})"

    def fit_linear_polynomial(self):
        """ Fits a linear polynomial to the x and y coordinates and returns a 
            vector of coefficients which describe the slope and y-intercept.
        """
        parameters = np.polyfit((self.pt1.x, self.pt2.x),
                                (self.pt1.y, self.pt2.y), 1)
        slope = parameters[0]
        y_intercept = parameters[1]
        return slope, y_intercept

    def is_horizontal(self, threshold: int = 0.01) -> bool:
        """ Check if a line is horizonal.

            Inputs:
                threshold: acceptable threshold from 0 to be considered horizontal

            Outputs:
                boolean
        """
        # the slope of a horizontal line is zero
        if abs(self.slope) <= threshold:
            return True
        return False

    def is_within_lane_slope_threshold(self, threshold: list) -> bool:
        """ Check to see if the line is within the lane slope threshold. 

            Inputs:
                threshold: acceptable lane slope threshold

            Outputs:
                boolean
        """
        if threshold[0] <= abs(self.slope) <= threshold[1]:
            return True
        return False

    def get_point_at_x(self, x: int) -> Point:
        """ Get (x,y) coordinated of point on the line at a given x location.

            Inputs:
                x: x coordinate
        """
        # y = mx+b
        y = self.slope*x + self.y_intercept
        return Point(x, y)

    def get_point_at_y(self, y: int) -> Point:
        """ Get (x,y) coordinated of point on the line at a given y location

            Inputs: 
                y: y coordinate
        """
        # x = (y-b)/m
        x = (y-self.y_intercept)/self.slope
        return Point(x, y)

    def draw(self, frame):
        """ Draw line on the input image frame

            Inputs:
                frame: image to draw the line on
        """
        # catch outlier lines that might overload the visualiztion function
        if abs(self.pt1.x) > 10000 or abs(self.pt1.y) > 10000:
            return
        else:
            cv2.line(frame, (self.pt1.x, self.pt1.y), (self.pt2.x,
                                                       self.pt2.y), (0, 255, 255), 3, cv2.LINE_AA)


class HoughLineDetection:
    """ Class for running OpenCV's hough line detection algorithm
    """

    def __init__(self) -> None:
        self.polygon_pts = []

        # Polygon that create a mask of the lane. To get a new set of points for
        # the lane mask, run the lane mask calibration in lane_following.py
        self.lane_mask = np.array(
            [[(236, 224), (4, 461), (4, 663), (975, 663), (973, 410), (784, 258)]])
        # self.lane_mask = np.array([[(253, 265), (8, 511), (8, 759), (1020, 757), (1018, 499), (729, 280)]])

    def do_canny_edge_detection(self, img: list) -> list:
        """ Apply OpenCV's canny edge detection to find the edges in the image

            Inputs:
                    img: cv image
        """
        # Converts frame to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Apply the gaussian noise kernel to smooth the image.
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        # Run canny
        canny = cv2.Canny(blur, 50, 100)
        return canny

    def mask_for_lane_region(self, img: list) -> list:
        """ Use a polygon to segment out the road portion of the image.

            Inputs:
                img: cv image
        """
        # Creates mask with only the area within the polygon filled with values
        # of 1 and the other areas filled in with 0
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, self.lane_mask, 255)
        # Filter for segment of the image that is within the mask.
        segment = cv2.bitwise_and(img, mask)
        return segment

    def do_hough_line_transform(self, img: list) -> list:
        """ Apply OpenCV's probablistic hough line transform to detect lines in 
            the image

            Inputs:
                img: cv image
        """
        canny = self.do_canny_edge_detection(img)
        segment = self.mask_for_lane_region(canny)
        lines = cv2.HoughLinesP(segment, 1, np.pi / 180, 100, None, 50, 10)

        return lines

    def draw_hough_lines(self, img: list, lines: list) -> list:
        """ Draw the predicted hough lines on the input image.

            Inputs:
                img: cv image
                lines: hough lines
        """
        if lines is not None and img is not None:
            for i in range(0, len(lines)):
                l = lines[i][0]
                cv2.line(img, (l[0], l[1]), (l[2], l[3]),
                         (0, 0, 255), 3, cv2.LINE_AA)

    def draw_canny_edges(self, edges: list) -> list:
        """ Draw canny edges

            Inputs:
                edges: canny edges
        """
        cv2.imshow('edge', edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
