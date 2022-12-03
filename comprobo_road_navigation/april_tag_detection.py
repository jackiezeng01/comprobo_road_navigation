import cv2
import apriltag


# vid = cv2.VideoCapture(0)
# ret, frame = vid.read()
# frame = cv2.imread('sample_images/apriltags_30mm.png')
frame = cv2.imread('/home/melody/ros2_ws/src/comprobo_road_navigation/sample_images/tagsampler.png')

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
options = apriltag.DetectorOptions(families="tag36h11")
detector = apriltag.Detector(options)
results = detector.detect(gray)

# loop over the AprilTag detection results
for r in results:
	# extract the bounding box (x, y)-coordinates for the AprilTag
	# and convert each of the (x, y)-coordinate pairs to integers
	(ptA, ptB, ptC, ptD) = r.corners
	ptB = (int(ptB[0]), int(ptB[1]))
	ptC = (int(ptC[0]), int(ptC[1]))
	ptD = (int(ptD[0]), int(ptD[1]))
	ptA = (int(ptA[0]), int(ptA[1]))
	# draw the bounding box of the AprilTag detection
	cv2.line(frame, ptA, ptB, (0, 255, 0), 2)
	cv2.line(frame, ptB, ptC, (0, 255, 0), 2)
	cv2.line(frame, ptC, ptD, (0, 255, 0), 2)
	cv2.line(frame, ptD, ptA, (0, 255, 0), 2)
	# draw the center (x, y)-coordinates of the AprilTag
	(cX, cY) = (int(r.center[0]), int(r.center[1]))
	cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
	# draw the tag family on the image
	tagFamily = r.tag_family.decode("utf-8")
	tagID = r.tag_id
	cv2.putText(frame, str(tagID), (ptA[0], ptA[1] - 15),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
# show the output image after AprilTag detection
cv2.imshow("Video frame", frame)
cv2.waitKey(0)