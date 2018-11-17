import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
dirname = os.path.dirname(__file__)


def find_diff(valid,invalid):

    im2, contours_valid, hierarchy = cv2.findContours(valid,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    im2, contours_invalid, hierarchy = cv2.findContours(invalid,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    dc = cv2.convexHull(contours_valid[0])
    areadiff1 = cv2.contourArea(contours_valid[0]) - cv2.contourArea(dc)
    dc2 = cv2.convexHull(contours_invalid[0])
    areadiff2 = cv2.contourArea(contours_invalid[0]) - cv2.contourArea(dc2)

    return abs(areadiff1)-abs(areadiff2), areadiff1, areadiff2

valids = []
invalids = []

for valid_num in range(1,155):
    valid = cv2.imread("/afs/inf.ed.ac.uk/user/s16/s1614973/Documents/IVR/IVR-assignment/valid/validxy%s.jpg"%valid_num,0)
    ret,thresh = cv2.threshold(valid,127,255,cv2.THRESH_BINARY)
    valids.append(thresh)

for invalid_num in range(1,157):
    invalid = cv2.imread("/afs/inf.ed.ac.uk/user/s16/s1614973/Documents/IVR/IVR-assignment/invalid/invalidxy%s.jpg"%invalid_num,0)
    ret,thresh = cv2.threshold(invalid,127,255,cv2.THRESH_BINARY)
    invalids.append(thresh)

smallestDiff = float('inf')
validNo = -1
validAreaDiff = 0
invalidNo = -1
invalidAreaDiff = 0
diffs = []

i = 1
for valid in valids:
    j = 1
    for invalid in invalids:
        print("On i=%s,j=%s"%(i,j))
        area_diffs, area_valid, area_invalid = find_diff(valid,invalid)
        cv2.waitKey(0)
        if area_diffs<smallestDiff:
            validNo = i
            invalidNo = j
            validAreaDiff = area_valid
            invalidAreaDiff = area_invalid
            smallestDiff = area_diffs
        diffs.append(area_diffs)
        j+=1
    i+=1
1
plt.hist(diffs,10)
plt.title("Histogram of how much greater difference between contour area and convex hull area is for each valid target compared to invalid")
plt.xlabel("Difference between valid and invalid targets difference between contour area and convex hull area")
plt.ylabel("Frequency")
plt.show()

print("Smallest diff is %s with valid %s and invalid %s with area diffs %s and %s respectively"
        %(smallestDiff,validNo,invalidNo,validAreaDiff,invalidAreaDiff))
