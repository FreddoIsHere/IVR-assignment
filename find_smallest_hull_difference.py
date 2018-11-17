import cv2
import os
import numpy as np
dirname = os.path.dirname(__file__)


def find_diff(valid,invalid):
    mask = np.concatenate((valid, invalid), axis=0)

    im2, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    dc = cv2.convexHull(contours[0])
    areadiff1 = cv2.contourArea(contours[0]) - cv2.contourArea(dc)
    dc2 = cv2.convexHull(contours[1])
    areadiff2 = cv2.contourArea(contours[1]) - cv2.contourArea(dc2)

    print("Area diffs %s and %s respectively"
            %(areadiff1,areadiff2))

    print("%s contours"%len(contours))

    return abs(areadiff2-areadiff1), areadiff1, areadiff2

valids = []
invalids = []

for valid_num in range(1,155):
    valid = cv2.imread("/afs/inf.ed.ac.uk/user/s16/s1614973/Documents/IVR/IVR-assignment/valid/validxy%s.jpg"%valid_num,0)
    valids.append(valid)

for invalid_num in range(1,157):
    invalid = cv2.imread("/afs/inf.ed.ac.uk/user/s16/s1614973/Documents/IVR/IVR-assignment/invalid/invalidxy%s.jpg"%invalid_num,0)
    invalids.append(invalid)

smallestDiff = float('inf')
validNo = -1
validAreaDiff = 0
invalidNo = -1
invalidAreaDiff = 0

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
        j+=1
    i+=1

print("Smallest diff is %s with valid %s and invalid %s with area diffs %s and %s respectively"
        %(smallestDiff,validNo,invalidNo,validAreaDiff,invalidAreaDiff))
