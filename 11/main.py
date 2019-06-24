import cv2
import numpy as np

cam = cv2.VideoCapture(0)

#define range of purple color in HSV
lower_purple = np.array([130, 50, 90])
upper_purple = np.array([170, 255, 255])

#create empty points array
points = []

#get default camera window size
ret, frame = cam.read()
height, width = frame.shape[:2]
frame_count = 0

while True:
    ret, frame = cam.read()
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv_img, lower_purple, upper_purple)
    _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create empty centre array to store centroid center of mass
    center = int(height/2), int(width/2)

    if len(contours) > 0:
        #Get the largest contour and its center 
        c = max(contours, key = cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        try:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        except:
            center =   int(height/2), int(width/2)

        if radius > 25:      
            #Draw cirlce and leave the last center creating a trail
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 0, 255), 2)
            cv2.circle(frame, center, 5, (0, 255, 0), -1)
            
    #Log center points 
    points.append(center)

    if radius > 25:
        for i in range(1, len(points)):
            try:
                cv2.line(frame, points[i - 1], points[i], (0, 255, 0), 2)
            except:
                pass
        frame_count = 0
    else:
        frame_count += 1
        
        if frame_count == 10:
            points = []
            frame_count = 0
            
    frame = cv2.flip(frame, 1)
    cv2.imshow("Object Tracker", frame)

    if cv2.waitKey(1) == 13:
        break

cam.release()
cv2.destroyAllWindows()