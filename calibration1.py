import cv2
from PIL import Image
import os
import PIL
import glob
import numpy as np
import matplotlib.pyplot as plt
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
CHECKERBOARD = (8,8)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*6,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
#objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
#objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
#prev_img_shape = None
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
print(objp)


i = 0
#img1 = cv2.imread("C:/Users/HP/PycharmProjects/pythonProject1/images/photo1.png")
#print(img1.shape)
#cv2.imshow("original image", img1)
print("praneeth")
#cv2.waitKey(0)
#cv2.destroyAllWindows()
images = glob.glob("C:/Users/HP/PycharmProjects/pythonProject1/images2/*jpg")
for fname in images:
    img = cv2.imread(fname)
    print("abc")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)
    #ret, corners = cv2.findChessboardCorners(
        #gray, CHECKERBOARD,
        #cv2.CALIB_CB_ADAPTIVE_THRESH
        #+ cv2.CALIB_CB_FAST_CHECK +
        #cv2.CALIB_CB_NORMALIZE_IMAGE)
    print("abc")
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)
        print("bde")
        print(fname)
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
        cv2.imshow('img',img)
        #cv2.waitKey(0)
    i=i+1
print(i)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
print(mtx)
print(dist)
print(rvecs)
i=0
img = cv2.imread("C:/Users/HP/PycharmProjects/pythonProject1/images2/Screenshot (69).jpg")
h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png',dst)
cv2.imshow("result is",dst)
#f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
#ax1.imshow(img)
#ax1.set_title('Original Image', fontsize=30)
#ax2.imshow(dst)
#ax2.set_title('Undistorted Image', fontsize=30)
cv2.waitKey(0)
plt.show()
mean_error = 0
print(objpoints)
n=0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    print(error)
    mean_error += error
    n=n+1
print(i)
print(len(objpoints))
print("total error: ", mean_error/len(objpoints))