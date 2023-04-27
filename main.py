import cv2
import numpy as np
from sift import sift_features
from ransac import ransac_homography
from homography import refine_homography
from mosaicing import mosaicing_images

# Load images
im1 = cv2.imread('1.jpg')
im2 = cv2.imread('2.jpg')
im3 = cv2.imread('3.jpg')
im4 = cv2.imread('4.jpg')
im5 = cv2.imread('5.jpg')

# Compute SIFT features
keypoints1, descriptors1 = sift_features(im1)
keypoints2, descriptors2 = sift_features(im2)

# Perform feature matching with RANSAC
matches = cv2.BFMatcher().match(descriptors1, descriptors2)
points1 = np.array([keypoints1[m.queryIdx].pt for m in matches])
points2 = np.array([keypoints2[m.trainIdx].pt for m in matches])
H = ransac_homography(points1, points2)

# Refine homography with nonlinear least squares
H = refine_homography(points1, points2, H)

# Mosaicing images
mosaic = mosaicing_images([im1, im2, im3, im4, im5], [H, H, np.eye(3), H, H])

# Display result
cv2.imshow('Mosaic', mosaic)
cv2.waitKey(0)
cv2.destroyAllWindows()
