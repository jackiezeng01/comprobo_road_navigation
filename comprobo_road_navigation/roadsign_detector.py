import cv2

class RoadSignDetector():

    def __init__(self):
        """ Initialize the road sign detector """
        self.cv_image = None
        self.road_sign_id_confidence = 0
        self.signs_detected = []
        self.cX = 0
        self.cY = 0
        
        # color mask bounds for road signs
        self.raw_cv_image = None
        self.sign_r_lower_bound = 0
        self.sign_r_upper_bound = 160
        self.sign_g_lower_bound = 127
        self.sign_g_upper_bound = 225
        self.sign_b_lower_bound = 0
        self.sign_b_upper_bound = 80
        self.shape_epsilon = 4

        # color mask bounds for traffic light green
        self.trafficlight_r_lower_bound = 0
        self.trafficlight_r_upper_bound = 8
        self.trafficlight_g_lower_bound = 42
        self.trafficlight_g_upper_bound = 92
        self.trafficlight_b_lower_bound = 0
        self.trafficlight_b_upper_bound = 46

    def reset_detector(self):
        """ Reset list of signs detected and confidence.
        """
        self.road_sign_id_confidence = 0
        self.signs_detected = []

    def detect_shape(self, c):
        """ Compute perimeter of contour and perform contour approximation.
             
            Args:
                c: contour to approximate into a polygon shape
        """
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
        """ Identify the contours detected in the color masked binary image.
        """
        cnts = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        return cnts

    def id_and_draw_shapes(self, cnts):
        """ Identify the polygon shapes from the contours in view, draw them on the
            video window and the determined shape to the list of signs detected.

            Args:
                cnts: list of contours detected in the color masked binary image.
        """
        max_size_so_far = 0
        roadsign = None
        for c in cnts:
            # Identify shape
            size = cv2.contourArea(c)
            if  size > 150:
                shape = self.detect_shape(c)
                # Find centroid and label shape name
                M = cv2.moments(c)
                if M["m00"] != 0:
                    self.cX = int(M["m10"] / M["m00"])
                    self.cY = int(M["m01"] / M["m00"])
                if size > max_size_so_far:
                    roadsign = shape
        cv2.drawContours(self.cv_image, cnts, -1, (0,255,0), 3)
        self.signs_detected.append(roadsign)

    def run_roadsign_detector(self, cv_image, raw_cv_image):
        """ Lane follower main function. Runs lane followuing behavior

        Args:
            cv_image: cv image to draw on
            raw_cv_image: undistorted cv image

        Returns:
            sign: str of the detected roadsign, None if no sign is detected
            self.cv_image: cv image with visualizations added
        """
        self.cv_image = cv_image
        self.raw_cv_image = raw_cv_image
        self.binary_image = cv2.inRange(self.raw_cv_image, (self.sign_b_lower_bound,self.sign_g_lower_bound,self.sign_r_lower_bound), (self.sign_b_upper_bound,self.sign_g_upper_bound,self.sign_r_upper_bound))
        contours = self.get_contours(self.binary_image)
        self.id_and_draw_shapes(contours)
        sign = self.id_sign_with_confidence()
        if self.confidence > 0.5:
            cv2.putText(self.cv_image, sign, (self.cX - 20, self.cY), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (36,4,12), 2)
            return sign, self.cv_image
        else:
            return None, self.cv_image

    def id_sign_with_confidence(self):
        """
        Add detected roadsign to list of signs detected
        Use this list to see which road sign it probably is (whichever has the highest count)
        and confidence (count of the road sign with the highest count / total count)
        """
        roadsign = max(self.signs_detected, key=self.signs_detected.count)
        self.confidence = self.signs_detected.count(roadsign)/len(self.signs_detected)
        return roadsign


    def get_traffic_light_action(self, raw_cv_image):
        """
        If a green light is spotted through the color mask, return 'go'
        otherwise return 'stop' (yellow/red light)
        """
        traffic_image = cv2.inRange(raw_cv_image, (self.trafficlight_b_lower_bound,self.trafficlight_g_lower_bound,self.trafficlight_r_lower_bound), (self.trafficlight_b_upper_bound,self.trafficlight_g_upper_bound,self.trafficlight_r_upper_bound))
        contours = cv2.findContours(traffic_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.id_and_draw_shapes(contours)
        for c in contours:
            # if we see a decently large blob of green, then it's probably a green light
            if cv2.contourArea(c) > 55:
                return 'go'
        return 'stop'
