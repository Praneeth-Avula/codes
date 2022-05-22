# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

#
import cv2
from PIL import Image
import os
import PIL
import glob
from imutils.perspective import four_point_transform
import numpy as np

height = 800
width = 600
green = (0, 255, 0)



image = cv2.imread("images/IMG_2508.jpg")
print(image.shape)
cv2.imshow('Original imagesss:', image)
image = cv2.resize(image, (width, height))
orig_image = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert the image to gray scale
blur = cv2.GaussianBlur(gray, (5, 5), 0) # Add Gaussian blur
edged = cv2.Canny(blur, 75, 200) # Apply the Canny algorithm to find the edges

# Show the image and the edges
cv2.imshow('Original image:', image)
cv2.imshow('Edged:', edged)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# Show the image and all the contours
cv2.imshow("Image", image)
cv2.drawContours(image, contours, -1, green, 3)
cv2.imshow("All contours", image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
for contour in contours:
    # we approximate the contour
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.05 * peri, True)
    # if we found a countour with 4 points we break the for loop
    # (we can assume that we have found our document)
    if len(approx) == 4:
        doc_cnts = approx
        break

cv2.drawContours(orig_image, [doc_cnts], -1, green, 3)
cv2.imshow("Contours of the document", orig_image)
print(doc_cnts)
rows = len(doc_cnts)
column = len(doc_cnts[0])
x=sum(doc_cnts[0])
print(x)
print(rows)
print(column)
print(doc_cnts[0])
s = doc_cnts[0].sum(axis=0)
t = doc_cnts[1].sum(axis=0)
u = doc_cnts[2].sum(axis=0)
v = doc_cnts[3].sum(axis=0)
pts = [[s[0],s[1]],
       [t[0],t[1]],
       [u[0],u[1]],
       [v[0],v[1]]]
p=s[0]+s[1]
print(p)
#pts.sort(key=lambda x:x[0]+x[1])
pts = np.float32(pts)
rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
s = pts.sum(axis = 1)
rect[0] = pts[np.argmin(s)]
rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
diff = np.diff(pts, axis = 1)
rect[1] = pts[np.argmin(diff)]
rect[3] = pts[np.argmax(diff)]
print("the points are", rect)
#pts = ([doc_cnts[0]],[doc_cnts[1]],[doc_cnts[2]],[doc_cnts[3]])
#print(pts)
#figure = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
#s = pts.sum(axis=1)
#print(s)
#figure[0] = pts[np.argmin(s)]
#figure[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
#diff = np.diff(pts, axis = 1)
#figure[1] = pts[np.argmin(diff)]
#figure[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	# obtain a consistent order of the points and unpack them
	# individually
#figure = order_points(doc_cnts)
tl, tr, br, bl = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
maxWidth = max(int(widthA), int(widthB))
print(maxWidth)
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
dst = np.float32([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]])
print(dst)
	# compute the perspective transform matrix and then apply it
M = cv2.getPerspectiveTransform(rect,dst)
print(M)
warped = cv2.warpPerspective(orig_image, M, (maxWidth, maxHeight))
#out = cv2.warpPerspective(orig_image,M,(maxWidth, maxHeight),flags=cv2.INTER_LINEAR)
#warped = cv2.warpPerspective(image,M,(maxWidth, maxHeight))
#warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
cv2.imshow("Scanned",warped)
#flipHorizontal = cv2.flip(warped, 1)
#cv2.imshow("Scanned image",flipHorizontal)
greyscale = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
#blurs = cv2.GaussianBlur(greyscale, (5, 5), 0)
#edgedes = cv2.Canny(blurs, 75, 200)
(thresh, binaryimage) = cv2.threshold(greyscale, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("final image",binaryimage)

cv2.waitKey(0)
#warped = four_point_transform(orig_image, doc_cnts.reshape(4, 2))
# convert the warped image to grayscale

#cv2.imshow("Scanned", cv2.resize(warped, (600, 800)))
#cv2.waitKey(0)


