import cv2
import numpy as np

def ORB_detector(image,template):
	img_1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	img_2 = template
	orb = cv2.ORB_create()
	
	# obtain descriptors
	(_, desc_1) = orb.detectAndCompute(img_1, None)
	(_, desc_2) = orb.detectAndCompute(img_2, None)
	
	# create matcher
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
	matches = bf.match(desc_1, desc_2)
	
	# sort the matches based on distancee. Least distance is better
	matches = sorted(matches, key = lambda val: val.distance)
	
	return len(matches)

# capturing input
cam = cv2.VideoCapture(0)		
target = cv2.imread('pringles.jpg',0)

while True:
	ret, frame = cam.read()
	height,width = frame.shape[:2]
	
	top_left_x = int(width / 3)
	top_left_y = int((height / 2) + (height / 4))
	bot_right_x = int((width / 3) * 2)
	bot_right_y = int((height /2 ) - (height / 4))
	
	# draw rec window
	cv2.rectangle(frame, (top_left_x, top_left_y), (bot_right_x, bot_right_y), 255, 3)
	
	#crop window
	cropped = frame[bot_right_y:top_left_y, top_left_x:bot_right_x]
	
	# flip frame orientation horizontally
	frame = cv2.flip(frame, 1)
	
	matches = ORB_detector(cropped,target)
	cv2.putText(frame, str(matches), (450, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 1)
	
	# indicate object detection
	threshold = 200
	
	#match if exceed threshold
	if matches > threshold:
		cv2.rectangle(frame, (top_left_x, top_left_y), (bot_right_x, bot_right_y), (0, 255, 0), 3)
		cv2.putText(frame, 'Object Found', (50,50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
		
	cv2.imshow('Object Detector', frame)
	if cv2.waitKey(1) == 13:
		break
		
cam.release()
cv2.destroyAllWindows()