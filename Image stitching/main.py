import numpy as np
import cv2

im1 = cv2.imread('og1.jpg')
im2 = cv2.imread('og2.jpg')

sift = cv2.ORB_create()
kp1, des1 = sift.detectAndCompute(im1, None)
kp2, des2 = sift.detectAndCompute(im2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

pts1 = []
pts2 = []
for mat in matches:
    pts1.append(kp1[mat.queryIdx].pt)
    pts2.append(kp2[mat.trainIdx].pt)
pts1 = np.asarray(pts1, dtype = np.float32)
pts2 = np.asarray(pts2, dtype = np.float32)

M_hom, inliers = cv2.findHomography(pts2, pts1, cv2.RANSAC)
pano_size = (int(M_hom[0, 2] + im2.shape[1]), max(im1.shape[0], im2.shape[0]))
img_pano = cv2.warpPerspective(im2, M_hom, pano_size)
img_pano[0:im1.shape[0], 0:im1.shape[1], :] = im1
cv2.imwrite('stitch.jpg', img_pano)