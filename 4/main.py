import cv2
import numpy as np

image = cv2.imread('scene.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
target = cv2.imread('waldo.jpg', 0)

result = cv2.matchTemplate(gray, target, cv2.TM_CCOEFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# create bounding box
top_left = max_loc
bot_right = (top_left[0] + 50, top_left[1] + 50)
cv2.rectangle(image, top_left, bot_right, (0, 0, 255), 5)

cv2.imshow('Find Waldo', image)
cv2.waitKey(0)
cv2.destroyAllWindows()