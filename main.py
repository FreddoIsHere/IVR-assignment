#!/usr/bin/env python2.7
import gym
import reacher3D.Reacher
import numpy as np
import cv2
import math
import scipy as sp
import collections
import time
class MainReacher():
    def __init__(self):
        self.env = gym.make('3DReacherMy-v0')
        self.env.reset()
        
    def get_illumination(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
        return (np.mean(img[:,:,0])/255)

    def detect_red(self, image): # xz-image
        mask = cv2.inRange(image, (20, 0, 0),(255, 0, 0))
        kernel = np.ones((5,5),np.uint8)
        mask = cv2.dilate(mask,kernel,iterations=2)
        mask=cv2.erode(mask,kernel,iterations=3)

        M = cv2.moments(mask)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        return self.coordinate_convert(np.array([cx,cy]))
    
    def detect_green(self, image):
        mask = cv2.inRange(image, (0, 20, 0),(0, 255, 0))
        kernel = np.ones((5,5),np.uint8)
        mask = cv2.dilate(mask,kernel,iterations=2)
        mask=cv2.erode(mask,kernel,iterations=3)

        M = cv2.moments(mask)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        return self.coordinate_convert(np.array([cx,cy]))
    
    def detect_blue(self, image, lumi):
        mask = cv2.inRange(image, (0,0,245*lumi),(0,0,255))
        kernel = np.ones((5,5),np.uint8)
        mask = cv2.dilate(mask,kernel,iterations=2)
        mask=cv2.erode(mask,kernel,iterations=3)

        M = cv2.moments(mask)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        return self.coordinate_convert(np.array([cx,cy]))
    
    def detect_end(self, image, lumi):
        mask = cv2.inRange(image, (0,0,5),(0,0,140*lumi))
        kernel = np.ones((5,5),np.uint8)
        mask = cv2.dilate(mask,kernel,iterations=2)
        mask=cv2.erode(mask,kernel,iterations=3)
        
        M = cv2.moments(mask)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        return self.coordinate_convert(np.array([cx,cy]))
    
    def detect_target(self, image, lumi):
        a = 140 *lumi
        b = 190 *lumi
        mask = cv2.inRange(image, (a,a,a),(b,b,b))
        im2, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        image = image.copy()
        
        dc = cv2.convexHull(contours[0])
        areadiff1 = cv2.contourArea(contours[0]) - cv2.contourArea(dc)
        dc2 = cv2.convexHull(contours[1])
        areadiff2 = cv2.contourArea(contours[1]) - cv2.contourArea(dc2)
        
        if areadiff2 < areadiff1:
            M = cv2.moments(contours[1])
        else:
            M = cv2.moments(contours[0])
            
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
            
        return self.coordinate_convert([cx, cy])
        
    def detect_target2(self, image, lumi):
        a = 140 *lumi
        b = 190 *lumi
        mask = cv2.inRange(image, (a,a,a),(b,b,b))
        #kernel = np.ones((5,5),np.uint8)
        #mask = cv2.dilate(mask,kernel,iterations=6)
        #mask= cv2.erode(mask,kernel,iterations=5)
        #dst =cv2.cornerHarris(mask,2,3,0.001)
        #mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        #mask[dst>0.01*dst.max()]=[0,0,255]
        #mask = cv2.inRange(mask, (0, 0, 0),(0, 0, 255))
        
        mask = 255 - cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        params = cv2.SimpleBlobDetector_Params()
        # Change thresholds
        params.minThreshold = 10;
        params.maxThreshold = 200;
 
        # Filter by Area.
        params.filterByArea = True
        params.minArea = 0.1
        params.maxArea = 1000
 
        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.1
 
        params.maxCircularity = 0.6
 
        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.2
 
        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.01
        
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(mask)
        print(self.average_keypoint(keypoints))
        print(self.env.ground_truth_valid_target)
        #im_with_keypoints = cv2.drawKeypoints(mask, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return self.average_keypoint(keypoints)
    
    def average_keypoint(self, keypoints):
        x = 0
        y = 0
        for k in keypoints:
            x += k.pt[0]
            y += k.pt[1]
        l = len(keypoints)
        return self.coordinate_convert([x/l, y/l])
            
    
    def coordinate_convert(self,pixels):
        #Converts pixels into metres
        return np.array([(pixels[0]-self.env.viewerSize/2)/self.env.resolution,-(pixels[1]-self.env.viewerSize/2)/self.env.resolution])
    
    def get_target_coords(self, xyarray, xzarray):
        lumi1 = self.get_illumination(xyarray)
        lumi2 = self.get_illumination(xzarray)
        xy = self.detect_target(xyarray, lumi1)
        xz = self.detect_target(xzarray, lumi2)
        avgx = (xy[0] + xz[0])/2
        
        return [avgx, xy[1], xz[1]]

    def go(self):
        #The robot has several simulated modes:
        #These modes are listed in the following format:
        #Identifier (control mode) : Description : Input structure into step function

        #POS : A joint space position control mode that allows you to set the desired joint angles and will position the robot to these angles : env.step((np.zeros(3),np.zeros(3), desired joint angles, np.zeros(3)))
        #POS-IMG : Same control as POS, however you must provide the current joint angles and velocities : env.step((estimated joint angles, estimated joint velocities, desired joint angles, np.zeros(3)))
        #VEL : A joint space velocity control, the inputs require the joint angle error and joint velocities : env.step((joint angle error (velocity), estimated joint velocities, np.zeros(3), np.zeros(3)))
        #TORQUE : Provides direct access to the torque control on the robot : env.step((np.zeros(3),np.zeros(3),np.zeros(3),desired joint torques))
        self.env.controlMode="POS"
        #Run 100000 iterations
        prev_JAs = np.zeros(3)
        prev_jvs = collections.deque(np.zeros(3),1)
    

        # Uncomment to have gravity act in the z-axis
        # self.env.world.setGravity((0,0,-9.81))

        for i in range(100000):
            #The change in time between iterations can be found in the self.env.dt variable
            dt = self.env.dt
            #self.env.render returns 2 RGB arrays of the robot, one for the xy-plane, and one for the xz-plane
            arrxy,arrxz = self.env.render('rgb-array')
            
            if i == 100:
                print(str(self.get_target_coords(arrxy, arrxz)))
                print(str(self.env.ground_truth_valid_target))
                #print(self.get_illumination(arrxy))
                #]self.detect_target2(arrxy, self.get_illumination(arrxy))
                #xy = self.detect_target(arrxy, self.get_illumination(arrxy))
                #cv2.imshow( "Display window", self.detect_target2(arrxy, self.get_illumination(arrxy)))
                #cv2.waitKey(0)

            desired_joint_angles = np.array([0.5, 0.5, 0.3, 0])
            # self.env.step((np.zeros(3),np.zeros(3),jointAngles, np.zeros(3)))
            #self.env.step((np.zeros(3),np.zeros(3),np.zeros(3), np.zeros(4)))
            self.env.step((np.zeros(3),np.zeros(3), desired_joint_angles, np.zeros(3)))
            #The step method will send the control input to the robot, the parameters are as follows: (Current Joint Angles/Error, Current Joint Velocities, Desired Joint Angles, Torque input) 

#main method
def main():
    reach = MainReacher()
    reach.go()

if __name__ == "__main__":
    main()