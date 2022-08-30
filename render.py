import cv2
from PIL import Image
import os
import PIL
import glob
import numpy as np
import math
height =240
width = 360

image = cv2.imread("C:/Users/HP/PycharmProjects/pythonProject1/images/image002.png")
image = cv2.resize(image, (width, height))
#Intrinsic Matrix
mtx = np.array([[500,0,179.5],[0,500,119.5],[0,0,1]])
print( "Intrinsic matrix is:",mtx)
# Distortion Co-efficients
r1 = np.array([[-1,0,0,0,0]])
r1 = np.float32(r1)
print("The Distortion coefiicients:",r1)
# Position of Camera in world co-ordinates
tvecs = np.array([[2],[-2],[2]])
tvecs = np.float32(tvecs)
print("Position of Camera in world co-ordinates:",tvecs)
# Rotation Matrix in World co-ordinates
rvecs = np.array([[-0.36 , 0.8 , -0.48],[0.48,0.6,0.64],[0.8,0,-0.6]])
print("Rotation Matrix is:",rvecs)
x = np.array([[180.0],[120.0]])
print("Pixel value:",x)
p = np.int32(x)
image = cv2.circle(image, (p[0][0],p[1][0]), radius=3, color=(0, 0, 255), thickness=-1)
cv2.imshow("pic show",image)
cv2.waitKey(0)
#Undistorting the image
dst = cv2.undistortPoints(x,mtx,r1)
print("the value is:",dst)
s = dst[0].sum(axis=0)
print(s[0])
x1 = np.array([[s[0]],[s[1]],[1.0]])
print( "The Ray vector in Camera Coordinate system:",x1)
x1=np.float32(x1)
#Conversion of ray from Camera coordinate system to World Coordinate system
x5 = np.matmul(rvecs,x1)
print("The Ray vector in World Coordinate system:",x5)
x6 = np.reshape(x5,(1,3))
print("the value is :", x6)
x7 = x6[:,0][0]
x8 = x6[:,1][0]
x9 = x6[:,2][0]
#print(x7)

def LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint):

	ndotu = planeNormal.dot(rayDirection)
	w = rayPoint - planePoint
	t = -planeNormal.dot(w) / ndotu
	Point = rayPoint + t * rayDirection
	return Point

#Finding the Intersection point of Line with  XY plane
	#Define plane
planeNormal = np.array([0, 0, 1])
planePoint = np.array([0, 0, 0]) #Any point on the plane

	#Define ray
rayDirection = np.array([x7, x8, x9])
rayPoint = np.array([2, -2, 2]) #Any point along the ray


Point = LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint)
print ("intersection at", Point)
image1 = cv2.imread("C:/Users/HP/PycharmProjects/pythonProject1/images/testimage.png")
cv2.imshow("Test image",image1)
cv2.waitKey(0)