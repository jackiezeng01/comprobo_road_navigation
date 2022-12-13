import cv2

class RoadSignDetector():

    def __init__(self):
        """ Initialize the road sign detector """
        self.cv_image = None
        self.raw_cv_image = None
        self.red_lower_bound = 0
        self.red_upper_bound = 160
        self.green_lower_bound = 127
        self.green_upper_bound = 225
        self.blue_lower_bound = 0
        self.blue_upper_bound = 80
        self.shape_epsilon = 4

    def detect_shape(self, c):
        # Compute perimeter of contour and perform contour approximation
        sign = ""
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, self.shape_epsilon/100 * peri, True)
        num_edges = len(approx)
        if num_edges == 3: # Triangle
            sign = f"Yield"
        elif num_edges == 4: # Square or rectangle
            sign = f"Traffic Light Ahead"
        elif num_edges >= 5 and num_edges <= 8:
            sign = f"Stop"
        # Otherwise assume as circle or oval
        else:
            sign = f"Do Not Enter"
        return sign
        
    def get_contours(self, binary_image):
        # Find contours and detect shape
        cnts = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        return cnts

    def id_and_draw_shapes(self, cnts):
        shapes = []
        for c in cnts:
            # Identify shape
            if cv2.contourArea(c) > 55:
                shape = self.detect_shape(c)
                # Find centroid and label shape name
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.putText(self.cv_image, shape, (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (36,4,12), 2)
                shapes.append(shape)
        cv2.drawContours(self.cv_image, cnts, -1, (0,255,0), 3)
        return shapes

    def run_roadsign_detector(self, cv_image, raw_cv_image):
        self.cv_image = cv_image
        self.raw_cv_image = raw_cv_image
        self.binary_image = cv2.inRange(self.raw_cv_image, (self.blue_lower_bound,self.green_lower_bound,self.red_lower_bound), (self.blue_upper_bound,self.green_upper_bound,self.red_upper_bound))
        contours = self.get_contours(self.binary_image)
        shapes = self.id_and_draw_shapes(self.cv_image, contours)
        return shapes

    def get_traffic_light_action(self, raw_cv_image):
        return 'stop'
        # color mask, if spot green return 'go'
        # otherwise return 'stop'