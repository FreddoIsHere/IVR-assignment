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
        mask = cv2.inRange(image, (0,0,200*lumi),(0,0,255))
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
    
    def detect_joint_angles(self, xyarray, xzarray):
        lumi1 = self.get_illumination(xyarray)
        lumi2 = self.get_illumination(xzarray)
        
        redxz = self.detect_red(xzarray)
        redxy = self.detect_red(xyarray)
        ja1 = math.atan2(redxz[1],redxz[0])
        
        greenxy = self.detect_green(xyarray)
        ja2 = math.atan2(greenxy[1]-redxy[1],greenxy[0]-redxy[0])
        ja2 = self.angle_normalize(ja2)
        
        bluexy = self.detect_blue(xyarray, lumi1)
        ja3 = math.atan2(bluexy[1]-greenxy[1],bluexy[0]-greenxy[0])-ja2
        ja3 = self.angle_normalize(ja3)
        
        endxz = self.detect_end(xzarray, lumi2)
        
        ja4 = math.atan2(endxz[1]-redxz[1],endxz[0]-redxz[0])-ja1
        ja4 = self.angle_normalize(ja4)
        
        print(str([ja1, ja2, ja3, ja4]))
        
        return [ja1, ja2, ja3, ja4]
    
    def angle_normalize(self,x):
        #Normalizes the angle between pi and -pi
        return (((x+np.pi) % (2*np.pi)) - np.pi)
    
    #new functions
    def link_transform_z(self,angle):
        #Calculate the Homogenoeous transformation matrix from rotation and translation
        rot = np.matrix([[np.cos(angle), -np.sin(angle), 0, 0],
                        [np.sin(angle), np.cos(angle), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        trans = np.matrix(np.eye(4, 4))
        trans[0, 3] = 1
        return rot*trans
    
    def link_transform_y(self,angle):
        #Calculate the Homogenoeous transformation matrix from rotation and translation
        rot = np.matrix([[np.cos(angle), 0, np.sin(angle), 0],
                        [0, 1, 0, 0],
                        [-np.sin(angle), 0, np.cos(angle), 0],
                        [0, 0, 0, 1]])
        trans = np.matrix(np.eye(4, 4))
        trans[0, 3] = 1
        return rot*trans
    
    def Jacobian(self,joint_angles):
        #Forward Kinematics to calculate end effector location
        jacobian = np.zeros((6,4))
        z_vector = np.array([0,0,1])
        y_vector = np.array([0, 1, 0])
        
        j1_trans = self.link_transform_y(joint_angles[0])
        j2_trans = self.link_transform_z(joint_angles[1])
        j3_trans = self.link_transform_z(joint_angles[2])
        j4_trans = self.link_transform_y(joint_angles[3])
        
        ee_pos = (j1_trans*j2_trans*j3_trans*j4_trans)[0:3, 3].flatten()
        j4_pos = (j1_trans*j2_trans*j3_trans)[0:3, 3].flatten()
        j3_pos = (j1_trans*j2_trans)[0:3, 3].flatten()
        j2_pos = (j1_trans)[0:3, 3].flatten()
        j1_pos = np.array([0, 0, 0])
        
        pos3D = np.array([0, 0, 0])
        pos3D = (ee_pos-j1_pos)
        jacobian[0:3, 0] = np.cross(y_vector, pos3D)
        pos3D[0:3] = (ee_pos-j2_pos)
        jacobian[0:3, 1] = np.cross(z_vector, pos3D)
        pos3D[0:3] = (ee_pos-j3_pos)
        jacobian[0:3, 2] = np.cross(z_vector, pos3D)
        pos3D[0:3] = (ee_pos-j4_pos)
        jacobian[0:3, 3] = np.cross(y_vector, pos3D)
        jacobian[3:6, 0] = y_vector
        jacobian[3:6, 1] = z_vector
        jacobian[3:6, 2] = z_vector
        jacobian[3:6, 3] = y_vector
        
        return jacobian
    
    def IK(self, current_joint_angles, desired_position):
        
        curr_pos = self.FK(current_joint_angles)[0:3,3]
        pos_error = desired_position - np.squeeze(np.array(curr_pos.T))
        
        Jac = np.matrix(self.Jacobian(current_joint_angles))[0:3, :]
        
        if np.linalg.det(Jac*Jac.T) == 0:
            Jac_inv = Jac.T
        else:
            Jac_inv = Jac.T*np.linalg.inv(Jac*Jac.T)
        
        q_dot = Jac_inv*np.matrix(pos_error).T
        
        return np.squeeze(np.array(q_dot.T))
    
    def FK(self,joint_angles):
        #Forward Kinematics to calculate end effector location
        #Each link is 1m long
        #Calculate transformation matrix of each link
        j1_trans = self.link_transform_y(joint_angles[0])
        j2_trans = self.link_transform_z(joint_angles[1])
        j3_trans = self.link_transform_z(joint_angles[2])
        j4_trans = self.link_transform_y(joint_angles[3])
        #Combine transformation matrices
        total_transform = j1_trans*j2_trans*j3_trans*j4_trans
        
        return total_transform
        

    def go(self):
        #The robot has several simulated modes:
        #These modes are listed in the following format:
        #Identifier (control mode) : Description : Input structure into step function

        #POS : A joint space position control mode that allows you to set the desired joint angles and will position the robot to these angles : env.step((np.zeros(3),np.zeros(3), desired joint angles, np.zeros(3)))
        #POS-IMG : Same control as POS, however you must provide the current joint angles and velocities : env.step((estimated joint angles, estimated joint velocities, desired joint angles, np.zeros(3)))
        #VEL : A joint space velocity control, the inputs require the joint angle error and joint velocities : env.step((joint angle error (velocity), estimated joint velocities, np.zeros(3), np.zeros(3)))
        #TORQUE : Provides direct access to the torque control on the robot : env.step((np.zeros(3),np.zeros(3),np.zeros(3),desired joint torques))
        self.env.controlMode="VEL"
        #Run 100000 iterations
        prev_JAs = np.zeros(3)
        prev_jvs = collections.deque(np.zeros(3),1)
        
        prevJointAngles = np.zeros(4)
    

        # Uncomment to have gravity act in the z-axis
        # self.env.world.setGravity((0,0,-9.81))

        for i in range(100000):
            #The change in time between iterations can be found in the self.env.dt variable
            dt = self.env.dt
            #self.env.render returns 2 RGB arrays of the robot, one for the xy-plane, and one for the xz-plane
            arrxy,arrxz = self.env.render('rgb-array')            
         
            detectedJointAngles = self.env.ground_truth_joint_angles
            # Get current position of target
            ee_target = self.env.ground_truth_valid_target

            #Get the angles required using IK
            jointAngles = self.IK(detectedJointAngles,ee_target)
            
            detectedJointVels = self.angle_normalize(detectedJointAngles-prevJointAngles)/dt
            
            prevJointAngles = detectedJointAngles
            
            self.env.step((jointAngles , detectedJointVels , np.zeros(3), np.zeros(3)))
            
def main():
    reach = MainReacher()
    reach.go()

if __name__ == "__main__":
    main()