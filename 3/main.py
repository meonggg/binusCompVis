import cv2
import numpy as np

img_1 = cv2.imread('test_2.png', 0)
img_2 = cv2.imread('circles_only.png', 0)

detector = cv2.SimpleBlobDetector_create()

#detect blobs
keypoints_1 = detector.detect(img_1)
keypoints_2 = detector.detect(img_2)

#draw blobs as red circle
blank = np.zeros((1, 1))
blobs_1 = cv2.drawKeypoints(img_1, keypoints_1, blank, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
blobs_2 = cv2.drawKeypoints(img_2, keypoints_2, blank, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

num_blobs = len(keypoints_1)
text_1 = "Total Blobs on Img 1: " + str(len(keypoints_1))
cv2.putText(blobs_1, text_1, (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 255), 2)
text_2 = "Total Blobs on Img 2: " + str(len(keypoints_2))
cv2.putText(blobs_2, text_2, (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 255), 2)

cv2.imshow('Blobs 1', blobs_1)
cv2.imshow('Blobs 2', blobs_2)
cv2.waitKey(0)

cv2.destroyAllWindows()