import cv2 as cv
import numpy as np

image = cv.imread('inputs.png')
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

ret, thresh = cv.threshold(gray, 55, 255, 1)

_, contours, hierarchy = cv.findContours(thresh.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

for cnt in contours:
	approx = cv.approxPolyDP(cnt, 0.01 * cv.arcLength(cnt, True), True)
	if len(approx) == 3:
		shape_name = "Triangle"
		cv.drawContours(image, [cnt], 0, (0,255,0), -1)
		
		M = cv.moments(cnt)
		cx = int(M['m10'] / M['m00'])
		cy = int(M['m01'] / M['m00'])
		cv.putText(image, shape_name, (cx-50, cy), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
		
	elif len(approx) == 4:
		x, y, w, h = cv.boundingRect(cnt)
		M = cv.moments(cnt)
		cx = int(M['m10'] / M['m00'])
		cy = int(M['m01'] / M['m00'])
		
		if abs(w-h) < 3:
			shape_name = "Square"
			cv.drawContours(image, [cnt], 0, (0, 125, 255), -1)
			cv.putText(image, shape_name, (cx-50, cy), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
		else:
			shape_name = "Rectangle"
			cv.drawContours(image, [cnt], 0, (0 ,0 ,255), -1)
			M = cv.moments(cnt)
			cx = int(M['m10'] / M['m00'])
			cy = int(M['m01'] / M['m00'])
			cv.putText(image, shape_name, (cx-50, cy), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
			
	elif len(approx) == 10:
		shape_name = "Star"
		cv.drawContours(image, [cnt], 0, (255, 255, 0), -1)
		M = cv.moments(cnt)
		cx = int(M['m10'] / M['m00'])
		cy = int(M['m01'] / M['m00'])
		cv.putText(image, shape_name, (cx-50, cy), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
	
	elif len(approx) >= 15:
		shape_name = "Circle"
		cv.drawContours(image, [cnt], 0, (255, 255, 0), -1)
		M = cv.moments(cnt)
		cx = int(M['m10'] / M['m00'])
		cy = int(M['m01'] / M['m00'])
		cv.putText(image, shape_name, (cx-50, cy), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
		
cv.imshow('Shape Detection',image)
cv.waitKey(0)
cv.destroyAllWindows()