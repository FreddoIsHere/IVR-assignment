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
        #mask = cv2.dilate(mask,kernel,iterations=2)
        #mask=cv2.erode(mask,kernel,iterations=3)

        M = cv2.moments(mask)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            # We use matching (as opposed to moments) with the template to improve
            # approximating centre of circle when it is eclipsed by another joint
            # Use sub_mask to minimize search space, saving time
            lower_x = cx-51
            if lower_x<0:
                lower_x = 0
            lower_y = cy-51
            if lower_y<0:
                lower_y = 0
            sub_mask = mask[lower_y:cy+51,lower_x:cx+51]
            res = cv2.matchTemplate(sub_mask,self.red_temp,cv2.TM_CCOEFF)
            _, _, _, loc = cv2.minMaxLoc(res)
            # Add lower_x for offset caused by submask, add 26 to get to centre of template
            prev = [cx,cy]
            cx = loc[0]+lower_x+26
            cy = loc[1]+lower_y+26
        else:
            print("Got zero division error for red")
            return None

        return self.coordinate_convert(np.array([cx,cy]))

    def detect_green(self, image):
        mask = cv2.inRange(image, (0, 20, 0),(0, 255, 0))
        kernel = np.ones((5,5),np.uint8)
        #mask = cv2.dilate(mask,kernel,iterations=2)
        #mask=cv2.erode(mask,kernel,iterations=3)

        M = cv2.moments(mask)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            if self.show:
                self.green_mask = mask
            lower_x = cx-45
            if lower_x<0:
                lower_x = 0
            lower_y = cy-45
            if lower_y<0:
                lower_y = 0
            sub_mask = mask[lower_y:cy+45,lower_x:cx+45]
            res = cv2.matchTemplate(sub_mask,self.green_temp,cv2.TM_CCOEFF)
            _, _, _, loc = cv2.minMaxLoc(res)
            prev = [cx,cy]
            cx = loc[0]+lower_x+23
            cy = loc[1]+lower_y+23
        else:
            print("Got zero division error for green")
            return None

        return self.coordinate_convert(np.array([cx,cy]))

    def detect_blue(self, image, lumi):
        mask = cv2.inRange(image, (0,0,200*lumi),(0,0,255))
        kernel = np.ones((5,5),np.uint8)
        #mask = cv2.dilate(mask,kernel,iterations=2)
        #mask=cv2.erode(mask,kernel,iterations=3)

        M = cv2.moments(mask)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            lower_x = cx-37
            if lower_x<0:
                lower_x = 0
            lower_y = cy-37
            if lower_y<0:
                lower_y = 0
            sub_mask = mask[lower_y:cy+37,lower_x:cx+37]
            res = cv2.matchTemplate(sub_mask,self.blue_temp,cv2.TM_CCOEFF)
            _, _, _, loc = cv2.minMaxLoc(res)
            prev = [cx,cy]
            cx = loc[0]+lower_x+19
            cy = loc[1]+lower_y+19
        else:
            print("Got zero division error for blue")
            return None

        return self.coordinate_convert(np.array([cx,cy]))

    def detect_end(self, image, lumi):
        mask = cv2.inRange(image, (0,0,5),(0,0,140*lumi))
        kernel = np.ones((5,5),np.uint8)
        #mask = cv2.dilate(mask,kernel,iterations=2)
        #mask=cv2.erode(mask,kernel,iterations=3)

        M = cv2.moments(mask)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            lower_x = cx-27
            if lower_x<0:
                lower_x = 0
            lower_y = cy-27
            if lower_y<0:
                lower_y = 0
            sub_mask = mask[lower_y:cy+27,lower_x:cx+27]
            res = cv2.matchTemplate(sub_mask,self.end_temp,cv2.TM_CCOEFF)
            _, _, _, loc = cv2.minMaxLoc(res)
            prev = [cx,cy]
            cx = loc[0]+lower_x+14
            cy = loc[1]+lower_y+14
            print("Endxz: %s, expected: %s"%([cx,cy],prev))
            #cv2.imshow("End",mask)
            #cv2.waitKey(0)
        else:
            print("Got zero division error for end")
            return None

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

    def detect_joint_angles(self, xyarray, xzarray, prev_JAs, prev_jvs):
        lumi1 = self.get_illumination(xyarray)
        lumi2 = self.get_illumination(xzarray)

        self.prnt = True
        redxy = self.detect_red(xyarray)
        self.show = True
        greenxy = self.detect_green(xyarray)
        self.show = False
        bluexy = self.detect_blue(xyarray, lumi1)
        endxy = self.detect_end(xyarray, lumi1)
        self.prnt = False
        redxz = self.detect_red(xzarray)
        bluexz = self.detect_blue(xzarray, lumi2)
        endxz = self.detect_end(xzarray, lumi2)
        greenxz = self.detect_green(xzarray)

        if redxy is not None:
            redxy[1] = 0 # Know y coord should always be 0 as can't move in xy plane
            ja1_other = math.atan2(redxy[1],redxy[0])
            if self.prev_ja1_other is not None:
                self.prev_ja1_other_vel = self.angle_normalize(ja1_other-self.prev_ja1_other)/self.env.dt
            self.prev_ja1_other = ja1_other
        else:
            print("Estimated ja1_other")
            ja1_other = self.angle_normalize(self.prev_ja1_other+self.prev_ja1_other_vel*self.env.dt)

        if type(greenxy) == np.ndarray and type(redxy) == np.ndarray:
            ja2 = math.atan2(greenxy[1]-redxy[1],greenxy[0]-redxy[0])
            print("Raw ja2: %s"%ja2)
            if ja1_other != 0:
                ja2 = ja2*-1
            ja2 -= ja1_other
            ja2 = self.angle_normalize(ja2)
        else:
            print("Estimated ja2")
            ja2 = self.angle_normalize(prev_JAs[1]+prev_jvs[1]*self.env.dt)


        if type(bluexy) == np.ndarray and type(greenxy) == np.ndarray:
            ja3 = math.atan2(bluexy[1]-greenxy[1],bluexy[0]-greenxy[0])
            if ja1_other != 0:
                ja3 = ja3*-1
            print("Raw ja3: %s"%ja3)
            ja3 -= ja2 - ja1_other
            ja3 = self.angle_normalize(ja3)

        else:
            print("Estimated ja3")
            ja3 = self.angle_normalize(prev_JAs[2]+prev_jvs[2]*self.env.dt)

        if type(redxz) == np.ndarray:
            ja1 = math.atan2(redxz[1],redxz[0])
        else:
            print("Estimated ja1")
            ja1 = self.angle_normalize(prev_JAs[0]+prev_jvs[0]*self.env.dt)

        if type(endxz) == np.ndarray and type(bluexz) == np.ndarray:
            print("Ja1=%s"%ja1)
            print("Ja4 before Ja1=%s"%math.atan2(endxz[1]-bluexz[1],endxz[0]-bluexz[0]))
            ja4 = math.atan2(endxz[1]-bluexz[1],endxz[0]-bluexz[0])
            ja4 -= ja1
            ja4 = self.angle_normalize(ja4)
        else:
            print("Estimated ja4")
            ja4 = self.angle_normalize(prev_JAs[3]+prev_jvs[3]*self.env.dt)

        D = np.array([[1,0,0,-1],
                      [0,1,0,0],
                      [0,0,1,0],
                      [0,0,0,1]])
        v = np.matmul(D,self.rot_z(self.env.ground_truth_joint_angles[2]).T)
        v = np.matmul(v,D)
        v = np.matmul(v,self.rot_z(self.env.ground_truth_joint_angles[1]).T)
        v = np.matmul(v,D)
        v = np.matmul(v,self.rot_y(self.env.ground_truth_joint_angles[0]).T)
        v = np.matmul(v,np.array([endxz[0]-bluexz[0],
                                  endxz[1]-bluexz[1],
                                  endxy[1]-bluexy[1],
                                  0]).reshape((4,1)))
        ja4 = math.atan2(v[1],v[0])

        #print(str([ja1, ja2, ja3, ja4]))

        return np.array([ja1, ja2, ja3, ja4])

    def angle_normalize(self,x):
        #Normalizes the angle between pi and -pi
        return (((x+np.pi) % (2*np.pi)) - np.pi)

    #new functions
    def link_transform_z(self,angle):
        #Calculate the Homogenoeous transformation matrix from rotation and translation
        rot = self.rot_z(angle)
        trans = np.matrix(np.eye(4, 4))
        trans[0, 3] = 1
        return rot*trans

    def rot_z(self, angle):
        rot = np.matrix([[np.cos(angle), -np.sin(angle), 0, 0],
                        [np.sin(angle), np.cos(angle), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        return rot

    def link_transform_y(self,angle):
        #Calculate the Homogenoeous transformation matrix from rotation and translation
        rot = self.rot_y(angle)
        trans = np.matrix(np.eye(4, 4))
        trans[0, 3] = 1
        return rot*trans

    def rot_y(self, angle):
        rot = np.matrix([[np.cos(angle), 0, -np.sin(angle), 0],
                        [0, 1, 0, 0],
                        [np.sin(angle), 0, np.cos(angle), 0],
                        [0, 0, 0, 1]])
        return rot


    def Jacobian(self,joint_angles):
        jacobian = np.zeros((6,4))

        j1_trans = self.link_transform_y(joint_angles[0])
        j2_trans = self.link_transform_z(joint_angles[1])
        j3_trans = self.link_transform_z(joint_angles[2])
        j4_trans = self.link_transform_y(joint_angles[3])

        ee_pos = (j1_trans*j2_trans*j3_trans*j4_trans)[0:3, 3]
        j4_pos = (j1_trans*j2_trans*j3_trans)[0:3, 3]
        j3_pos = (j1_trans*j2_trans)[0:3, 3]
        j2_pos = (j1_trans)[0:3, 3]
        j1_pos = np.zeros((3,1))

        pos3D = np.zeros(3)

        pos3D = (ee_pos-j1_pos).T
        z0_vector = [0, -1, 0]
        jacobian[0:3, 0] = np.cross(z0_vector, pos3D)
        pos3D[0:3] = (ee_pos-j2_pos).T

        #z1_vector = (j1_trans*np.array([0, 0, 1, 0]).reshape(4,1))[0:3].T
        z1_vector = (self.rot_y(joint_angles[0])[0:3, 0:3]*np.array([0, 0, 1]).reshape(3,1)).T

        jacobian[0:3, 1] = np.cross(z1_vector, pos3D)
        pos3D[0:3] = (ee_pos-j3_pos).T

        z2_vector = (self.rot_y(joint_angles[0])[0:3, 0:3]*self.rot_z(joint_angles[1])[0:3, 0:3]*np.array([0, 0, 1]).reshape(3,1)).T

        jacobian[0:3, 2] = np.cross(z2_vector, pos3D)
        pos3D[0:3] = (ee_pos-j4_pos).T

        z3_vector = (self.rot_y(joint_angles[0])[0:3, 0:3]*self.rot_z(joint_angles[1])[0:3, 0:3]*self.rot_z(joint_angles[2])[0:3, 0:3]*np.array([0, -1, 0]).reshape(3,1))[0:3].T

        jacobian[0:3, 3] = np.cross(z3_vector, pos3D)

        jacobian[3:6, 0] = z0_vector
        jacobian[3:6, 1] = z1_vector
        jacobian[3:6, 2] = z2_vector
        jacobian[3:6, 3] = z3_vector


        return jacobian

    def IK(self, current_joint_angles, desired_position):

        curr_pos = self.FK(current_joint_angles)[0:3,3]
        pos_error = desired_position - np.squeeze(np.array(curr_pos.T))

        Jac = np.matrix(self.Jacobian(current_joint_angles))[0:3, :]

        if (np.linalg.matrix_rank(Jac,0.4)<3):
            Jac_inv = Jac.T
            #Jac_inv = np.linalg.pinv(Jac, rcond=0.99999)
        else:
            Jac_inv = Jac.T*np.linalg.inv(Jac*Jac.T)

        q_dot = Jac_inv*np.matrix(pos_error).T

        return np.squeeze(np.array(q_dot.T))

    def FK(self,j):
        j1_trans = self.link_transform_y(j[0])
        j2_trans = self.link_transform_z(j[1])
        j3_trans = self.link_transform_z(j[2])
        j4_trans = self.link_transform_y(j[3])

        total_transform = j1_trans*j2_trans*j3_trans*j4_trans
        #print(np.cos(j[0])+np.cos(j[1])*np.cos(j[0])+np.cos(j[2]+j[1])*np.cos(j[0])+np.cos(j[0])*np.cos(j[2]+j[1])*np.cos(j[3]))

        return total_transform


    def go(self):
        #The robot has several simulated modes:
        #These modes are listed in the following format:
        #Identifier (control mode) : Description : Input structure into step function

        #POS : A joint space position control mode that allows you to set the desired joint angles and will position the robot to these angles : env.step((np.zeros(3),np.zeros(3), desired joint angles, np.zeros(3)))
        #POS-IMG : Same control as POS, however you must provide the current joint angles and velocities : env.step((estimated joint angles, estimated joint velocities, desired joint angles, np.zeros(3)))
        #VEL : A joint space velocity control, the inputs require the joint angle error and joint velocities : env.step((joint angle error (velocity), estimated joint velocities, np.zeros(3), np.zeros(3)))
        #TORQUE : Provides direct access to the torque control on the robot : env.step((np.zeros(3),np.zeros(3),np.zeros(3),desired joint torques))

        #self.env.controlMode="POS-IMG"
        self.env.controlMode="VEL"

        #Run 100000 iterations
        prev_JAs = np.zeros(4)
        prev_jvs = np.zeros(4)


        # Uncomment to have gravity act in the z-axis
        # self.env.world.setGravity((0,0,-9.81))

        self.red_temp = np.zeros((52,52))
        self.red_temp = cv2.circle(self.red_temp,(26,26),25,1,-1).astype(np.uint8)
        self.green_temp = np.zeros((46,46))
        self.green_temp = cv2.circle(self.green_temp,(23,23),22,1,-1).astype(np.uint8)
        self.blue_temp = np.zeros((38,38))
        self.blue_temp = cv2.circle(self.blue_temp,(19,19),18,1,-1).astype(np.uint8)
        self.end_temp = np.zeros((28,28))
        self.end_temp = cv2.circle(self.end_temp,(14,14),13,1,-1).astype(np.uint8)

        self.prev_ja1_other = None
        self.prev_ja2_other = None
        self.prev_ja3_other = None

        x,y,z=(0,0,0)

        for i in range(100000):
            #The change in time between iterations can be found in the self.env.dt variable
            dt = self.env.dt
            #self.env.render returns 2 RGB arrays of the robot, one for the xy-plane, and one for the xz-plane
            arrxy,arrxz = self.env.render('rgb-array')

            #if i == 100:
                #print(str(self.get_target_coords(arrxy, arrxz)))
                #print(str(self.env.ground_truth_valid_target))
                #print(self.get_illumination(arrxy))
                #]self.detect_target2(arrxy, self.get_illumination(arrxy))
                #cv2.imshow( "Display window", self.detect_target2(arrxy, self.get_illumination(arrxy)))
                #cv2.waitKey(0)

            #if i == 400:
                #angles = self.detect_joint_angles(arrxy, arrxz)
                #print(self.env.ground_truth_joint_angles)

            #if i == 0:
                #detectedJointAngles = self.env.ground_truth_joint_angles

            #if i > 1:

                #detectedJointAngles = self.detect_joint_angles(arrxy, arrxz)

                #print("Actual target:"+str(self.env.ground_truth_valid_target))
                #print("Predicted target:"+str(self.get_target_coords(arrxy, arrxz)))

            if i == 0:
                detectedJointAngles = np.zeros(4)
                detectedJointVels = np.zeros(4)
            else:
                detectedJointAngles = self.detect_joint_angles(arrxy, arrxz, prev_JAs, prev_jvs)
                #detectedJointAngles[1] = self.env.ground_truth_joint_angles[1]
                #detectedJointAngles[2] = self.env.ground_truth_joint_angles[2]
                #detectedJointAngles[3] = self.env.ground_truth_joint_angles[3]
                detectedJointVels = self.angle_normalize(detectedJointAngles-prev_JAs)/dt
                x, y, z = self.get_target_coords(arrxy, arrxz)
                print("Difference of target from true position: (%s,%s,%s)"%(x-self.env.ground_truth_valid_target[0],y-self.env.ground_truth_valid_target[1],z-self.env.ground_truth_valid_target[2]))

            #if i == 100:
            #    cv2.imshow("xy",arrxy)
            #    cv2.waitKey(0)

            jointAngles = self.IK(detectedJointAngles,[x, y, z])

            print("Actual angles: %s" % self.env.ground_truth_joint_angles)
            print("Predicted angles %s" % detectedJointAngles)
            print("Actual velocity: %s" % self.env.ground_truth_joint_velocities)
            print("Predicted velocity: %s" % detectedJointVels)
            #print("Difference in actual and pred vel: %s " % (detectedJointVels-self.env.ground_truth_joint_velocities))
            #if self.green_mask is not None:
                #cv2.imshow("Green mask",self.green_mask)
                #cv2.imshow("Redxy",arrxy)
                #cv2.waitKey(0)
            print("------------------------------")
            #if (detectedJointAngles[1]<-0.719 or start):
            #    cv2.imshow('Nothing',np.zeros(5))
            #    cv2.waitKey(0)
            #    start = True
            print(i)
            #if i>160:
            #    time.sleep(1)
            desired_joint_angles = np.array([ math.pi/2,math.pi/2+1,0,1])
            # self.env.step((np.zeros(3),np.zeros(3),jointAngles, np.zeros(3)))
            #self.env.step((np.zeros(3),np.zeros(3),np.zeros(3), np.zeros(4)))
            #self.env.step((np.zeros(3),np.zeros(3), desired_joint_angles, np.zeros(3)))

            #self.env.step((detectedJointAngles, detectedJointVels, desired_joint_angles, np.zeros(4))) #POS-IMG
            self.env.step((jointAngles , detectedJointVels , np.zeros(3), np.zeros(3))) #VEL


            #The step method will send the control input to the robot, the parameters are as follows: (Current Joint Angles/Error, Current Joint Velocities, Desired Joint Angles, Torque input)

            prev_JAs = detectedJointAngles
            prev_jvs = detectedJointVels

#main method
def main():
    reach = MainReacher()
    reach.go()

if __name__ == "__main__":
    main()
