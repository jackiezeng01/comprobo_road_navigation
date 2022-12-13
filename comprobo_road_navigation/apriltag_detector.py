from comprobo_road_navigation.path_planning import PathPlanning
import cv2
import apriltag

class AprilTagDetector():

    def __init__(self):
        """ Initialize the apriltag detector """
        self.apriltag_stop_distances = {0: 2600,
                                        1: 2600,
                                        2: 2300,
                                        3: 2600,
                                        4: 2300,
                                        5: 2600,
                                        6: 2300,
                                        7: 2600,
                                        8: 2600}
        self.cv_image = None
        self.raw_cv_image = None

    def detect_apriltags(self, instruction):
        """
        Detects apriltags in the camera view and returns them as a
        dictionary with tag IDs as keys and sizes as values
        """
        gray = cv2.cvtColor(self.raw_cv_image, cv2.COLOR_BGR2GRAY)
        options = apriltag.DetectorOptions(families="tag36h11")
        detector = apriltag.Detector(options)
        results = detector.detect(gray)
        apriltags = {}

        # loop over the AprilTag detection results
        for r in results:
            tagID = r.tag_id
            # if this is the apriltag we are looking for, draw in green, otherwise yellow
            color = (0, 255, 0)
            if instruction[0] == tagID:
                color = (0, 255, 0)
            # extract the bounding box (x, y)-coordinates for the AprilTag
            (ptA, ptB, ptC, ptD) = r.corners
            ptB = (int(ptB[0]), int(ptB[1]))
            ptC = (int(ptC[0]), int(ptC[1]))
            ptD = (int(ptD[0]), int(ptD[1]))
            ptA = (int(ptA[0]), int(ptA[1]))
            # draw the bounding box of the AprilTag detection
            cv2.line(self.cv_image, ptA, ptB, color, 2)
            cv2.line(self.cv_image, ptB, ptC, color, 2)
            cv2.line(self.cv_image, ptC, ptD, color, 2)
            cv2.line(self.cv_image, ptD, ptA, color, 2)
            # draw the center (x, y)-coordinates of the AprilTag
            (cX, cY) = (int(r.center[0]), int(r.center[1]))
            cv2.circle(self.cv_image, (cX, cY), 5, color, -1)
            size = pow(int(ptB[0]) - int(ptA[0]), 2)
            cv2.putText(self.cv_image, str(tagID), (ptA[0], ptA[1] + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(self.cv_image, str(size), (ptB[0], ptB[1] - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            apriltags[tagID] = size
        return apriltags

    def run_apriltag_detector(self, cv_image, raw_cv_image, instruction):
        self.cv_image = cv_image
        self.raw_cv_image = raw_cv_image
        aprilTag_to_look_for = instruction[0]
        size_at_which_to_stop = self.apriltag_stop_distances.get(aprilTag_to_look_for)
        aprilTags = self.detect_apriltags(instruction)
        if aprilTag_to_look_for in aprilTags.keys():
            # if we have reached the apriltag we are looking, return 1
            if aprilTags.get(aprilTag_to_look_for) >= size_at_which_to_stop:
                return 1, self.cv_image
        return 0, self.cv_image
