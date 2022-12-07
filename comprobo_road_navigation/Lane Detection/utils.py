import cv2
import numpy as np


class Point:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
        self.xy = (int(x), int(y))

    def __str__(self):
        return f"({self.x},{self.y})"


class Line:
    def __init__(self, x1, y1, x2, y2) -> None:
        self.pt1 = Point(x1, y1)
        self.pt2 = Point(x2, y2)
        self.arr = [[x1, y1], [x2, y2]]
        self.slope, self.y_intercept = self.fit_linear_polynomial()

    def __str__(self):
        return f"({self.pt1},{self.pt2})"

    def fit_linear_polynomial(self):
        # Fits a linear polynomial to the x and y coordinates and returns a vector of coefficients which describe the slope and y-intercept
        parameters = np.polyfit((self.pt1.x, self.pt2.x),
                                (self.pt1.y, self.pt2.y), 1)
        slope = parameters[0]
        y_intercept = parameters[1]
        return slope, y_intercept

    def draw(self, frame):
        cv2.line(frame, (self.pt1.x, self.pt1.y), (self.pt2.x,
                 self.pt2.y), (0, 255, 255), 3, cv2.LINE_AA)

    def __str__(self):
        return f"[{self.pt1}, {self.pt2}]"


class HoughLineDetection:
    def __init__(self) -> None:
        pass

    def do_canny_edge_detection(self, img):
        """ Apply OpenCV's canny edge detection to find the edges in the image
        """
        # Converts frame to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Apply the gaussian noise kernel to smooth the image.
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        canny = cv2.Canny(blur, 50, 100)
        return canny

    def mask_for_road_region(self, img):
        """ Create a polygon segment out the road portion of the image. We then use this 
            to mask out the rest of the image and only focus on the road
        """
        height = img.shape[0]-100
        # Create a mask of the places
        polygons = np.array(
            [[(0, height), (1024, height), (607, 307), (400, 300)]])
        # Creates mask with only the area within the polygon filled with values of 1 and the other areas filled in with 0
        mask = np.zeros_like(self, img)
        cv2.fillPoly(mask, polygons, 255)
        # Filter for segment of the image that is within the mask.
        segment = cv2.bitwise_and(img, mask)
        return segment

    def do_hough_line_transform(self, img):
        """ Apply OpenCv's probablistic hough line transform to detect lines in the image
        """
        canny = self.do_canny_edge_detection(img)
        segment = self.mask_for_road_region(canny)
        lines = cv2.HoughLinesP(segment, 1, np.pi / 180, 100, None, 50, 10)
        return lines

    def draw_hough_lines(self, img, lines):
        """ Draw the predicted hough lines on the input image
        """
        if lines is not None and img is not None:
            for i in range(0, len(lines)):
                l = lines[i][0]
                cv2.line(img, (l[0], l[1]), (l[2], l[3]),
                         (0, 0, 255), 3, cv2.LINE_AA)
