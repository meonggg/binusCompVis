import cv2
import numpy as np 

def sketch(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to gray image
    img_gray_blur = cv2.GaussianBlur(img_gray, (5,5), 0) #apply gaussian blur
    canny_edges = cv2.Canny(img_gray_blur, 10, 70) #detect edge
    ret, mask = cv2.threshold(canny_edges, 20, 255, cv2.THRESH_BINARY_INV)
    
    return mask

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow('Live Sketch', sketch(frame))
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()