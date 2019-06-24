import cv2
import numpy as np


body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')
car_classifier = cv2.CascadeClassifier('haarcascade_car.xml')

##pedestrian detection
cam = cv2.VideoCapture('walking.avi')

#start when video is loaded
while cam.isOpened():
	ret, frame = cam.read()
	frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	#pass frame to classifier
	bodies = body_classifier.detectMultiScale(gray, 1.2, 3)
	
	#extract bounding vox for any bodies indentified
	for (x,y,w,h) in bodies:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
		cv2.imshow('Pedestrians', frame)
		
	if cv2.waitKey(1) == 13:
		break

cam.release()		
####

##cars detection
cam = cv2.VideoCapture('cars.avi')

while cam.isOpened():
	ret, frame = cam.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	car = car_classifier.detectMultiScale(gray, 1.3, 3)
	
	for (x,y,w,h) in car:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
		cv2.imshow('Cars', frame)
		
	if cv2.waitKey(1) == 13:
		break
		
cam.release()
cv2.destroyAllWindows();