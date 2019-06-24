import cv2
import numpy as np

image = cv2.imread('abraham.jpg')
marked_damages = cv2.imread('mask.jpg', 0)

#create a mask out of marked image by changing all colors 
#that are not white, to black
ret, thresh1 = cv2.threshold(marked_damages, 254, 255, cv2.THRESH_BINARY)

#dilate /make thicker the marks 
#since thresholding has narrowed it slightly
kernel = np.ones((7,7), np.uint8)
mask = cv2.dilate(thresh1, kernel, iterations = 1)
cv2.imwrite("abraham_mask.png", mask)

restored = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

cv2.imshow('Restored', restored)
cv2.imwrite('Final_Photo.png', restored)
cv2.waitKey(0)
cv2.destroyAllWindows()