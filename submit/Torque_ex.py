#!/usr/bin/env python2.7
import gym
import reacher3D.Reacher
#import reacher3D.Reacher_new
import numpy as np
import cv2
import math
import scipy as sp
import collections
import time
from scipy.spatial import distance
class MainReacher():
    def __init__(self):
        self.env = gym.make('3DReacherMy-v0')
        self.env.reset()

    def detect_red(self, image): # xz-image
        mask = cv2.inRange(image, (20, 0, 0),(255, 0, 0))

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
            cx = loc[0]+lower_x+26
            cy = loc[1]+lower_y+26
        else:
            return None

        return self.coordinate_convert(np.array([cx,cy]))

    def detect_green(self, image):
        mask = cv2.inRange(image, (0, 20, 0),(0, 255, 0))

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
            cx = loc[0]+lower_x+23
            cy = loc[1]+lower_y+23
        else:
            return None

        return self.coordinate_convert(np.array([cx,cy]))

    def detect_blue(self, image, lumi):
        mask = cv2.inRange(image, (0,0,200*lumi),(0,0,255))

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
            cx = loc[0]+lower_x+19
            cy = loc[1]+lower_y+19
        else:
            return None

        return self.coordinate_convert(np.array([cx,cy]))

    def detect_end(self, image, lumi):
        mask = cv2.inRange(image, (0,0,5),(0,0,200*lumi-1))

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
            cx = loc[0]+lower_x+14
            cy = loc[1]+lower_y+14
        else:
            return None

        return self.coordinate_convert(np.array([cx,cy]))

    def detect_target(self, image, lumi):
        a = 140 *lumi
        b = 190 *lumi
        mask = cv2.inRange(image, (a,a,a),(b,b,b))
        im2, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        image = image.copy()

        if (len(contours)>=2):
            dc = cv2.convexHull(contours[0])
            areadiff1 = cv2.contourArea(contours[0]) - cv2.contourArea(dc)
            dc2 = cv2.convexHull(contours[1])
            areadiff2 = cv2.contourArea(contours[1]) - cv2.contourArea(dc2)
        else:
            return self.prev_target

        if areadiff2 < areadiff1:
            M = cv2.moments(contours[1])
        else:
            M = cv2.moments(contours[0])

        if (M['m00']!=0):
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
        else:
            return self.prev_target

        target = self.coordinate_convert([cx, cy])
        self.prev_target = target

        return target
    #-----------NEW CODE------------------------------
    def detect_obstacles_on_plane(self, image):
        mask = cv2.inRange(image, (0,0,0),(0,0,0))
        _, contours, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        obstacles = []
        for cont in contours:
            M = cv2.moments(cont)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            points = self.coordinate_convert(np.array([cx,cy]))
            # Divide by self.env.resolution to get radius in coordinate form
            area = cv2.contourArea(cont)
            rad = np.sqrt(area/np.pi)/self.env.resolution
            # Excludes contours from indicator dots
            if rad >= 0.15:
                obstacles.append(np.array([points[0],points[1],rad]))
        return obstacles

    def detect_obstacles(self, xyarray, xzarray):
        xyobj = self.detect_obstacles_on_plane(xyarray)
        xzobj = self.detect_obstacles_on_plane(xzarray)
        x_xzobj = np.array([obj[0] for obj in xzobj])
        obsts = []
        for i in range(0,len(xyobj)):
            obj_xy = xyobj[i]
            best_obj_xz_idx = np.argmin(np.abs(x_xzobj-obj_xy[0]))
            x = np.mean([xzobj[best_obj_xz_idx][0],obj_xy[0]])
            y = xyobj[i][1]
            z = xzobj[best_obj_xz_idx][1]
            rad = np.max([xzobj[best_obj_xz_idx][2],obj_xy[2]])
            obsts.append((x,y,z,rad))
        return obsts
    #-----------END NEW CODE------------------------------

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

        D = np.array([[1,0,0,-1],
                      [0,1,0,0],
                      [0,0,1,0],
                      [0,0,0,1]])

        if type(redxz) == np.ndarray:
            ja1 = math.atan2(redxz[1],redxz[0])
        else:
            ja1 = self.angle_normalize(prev_JAs[0]+prev_jvs[0]*self.env.dt)

        if type(greenxy) == np.ndarray and type(redxy) == np.ndarray and type(greenxz) == np.ndarray and type(redxz) == np.ndarray:
            v = np.matmul(D,self.rot_y(ja1).T)
            v = np.matmul(v,np.array([greenxy[0]-redxy[0],
                                      greenxy[1]-redxy[1],
                                      greenxz[1]-redxz[1],
                                      0]).reshape((4,1)))
            ja2 = math.atan2(v[1],v[0])
        else:
            ja2 = self.angle_normalize(prev_JAs[1]+prev_jvs[1]*self.env.dt)


        if type(bluexy) == np.ndarray and type(greenxy) == np.ndarray and type(bluexz) == np.ndarray and type(greenxz) == np.ndarray:
            v = np.matmul(D,self.rot_z(ja2).T)
            v = np.matmul(v,D)
            v = np.matmul(v,self.rot_y(ja1).T)
            v = np.matmul(v,np.array([bluexy[0]-greenxy[0],
                                      bluexy[1]-greenxy[1],
                                      bluexz[1]-greenxz[1],
                                      0]).reshape((4,1)))
            ja3 = math.atan2(v[1],v[0])
        else:
            ja3 = self.angle_normalize(prev_JAs[2]+prev_jvs[2]*self.env.dt)

        if type(endxz) == np.ndarray and type(endxy) == np.ndarray and type(bluexz) == np.ndarray and type(bluexy) == np.ndarray:
            v = np.matmul(D,self.rot_z(ja3).T)
            v = np.matmul(v,D)
            v = np.matmul(v,self.rot_z(ja2).T)
            v = np.matmul(v,D)
            v = np.matmul(v,self.rot_y(ja1).T)
            v = np.matmul(v,np.array([endxz[0]-bluexz[0],
                                      endxy[1]-bluexy[1],
                                      endxz[1]-bluexz[1],
                                      0]).reshape((4,1)))
            ja4 = math.atan2(v[2],v[0])
        else:
            ja4 = self.angle_normalize(prev_JAs[3]+prev_jvs[3]*self.env.dt)

        return np.array([ja1, ja2, ja3, ja4])

    def get_illumination(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
        return (np.mean(img[:,:,0])/255)

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

    def angle_normalize(self,x):
        #Normalizes the angle between pi and -pi
        return (((x+np.pi) % (2*np.pi)) - np.pi)

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

        return total_transform

    def ts_pd_control(self, c_ee_pos , c_ee_vel , d_ee_pos):

        P = np.array([60,60, 60])
        D = np.array([20,20, 20])
        P_error = np.matrix(d_ee_pos -c_ee_pos).T
        D_error = np.zeros(shape=(3,1)) - np.matrix(c_ee_vel).T

        PD_error = np.diag(P)*P_error + np.diag(D)*D_error

        return PD_error

    def grav(self, joint_angles):
        # Gravitational acceleration and mass
        g = 9.81
        m = 1

        j1_trans = self.link_transform_y(joint_angles[0])
        j2_trans = self.link_transform_z(joint_angles[1])
        j3_trans = self.link_transform_z(joint_angles[2])
        j4_trans = self.link_transform_y(joint_angles[3])

        ee_pos = (j1_trans*j2_trans*j3_trans*j4_trans)[0:3, 3]
        j4_pos = (j1_trans*j2_trans*j3_trans)[0:3, 3]
        j3_pos = (j1_trans*j2_trans)[0:3, 3]
        j2_pos = (j1_trans)[0:3, 3]
        j1_pos = np.zeros((3,1))

        p14 = ((ee_pos - j4_pos)/2 + j4_pos).flatten()
        p13 = ((j4_pos - j3_pos)/2 + j3_pos).flatten()
        p12 = ((j3_pos - j2_pos)/2 + j2_pos).flatten()
        p11 = ((j2_pos - j1_pos)/2 + j1_pos).flatten()

        p24 = ((ee_pos - j4_pos)/2 + j4_pos -j2_pos).flatten()
        p23 = ((j4_pos - j3_pos)/2 + j3_pos -j2_pos).flatten()
        p22 = ((j3_pos - j2_pos)/2 + j2_pos -j2_pos).flatten()

        p34 = ((ee_pos - j4_pos)/2 + j4_pos - j3_pos).flatten()
        p33 = ((j4_pos - j3_pos)/2 + j3_pos - j3_pos).flatten()

        p44 = ((ee_pos - j4_pos)/2).flatten()


        grav = np.array([0, 0, -m*g])

        z0 = np.array((self.rot_y(joint_angles[0])[0:3, 0:3] * np.array([0, -1, 0]).reshape(3,1)).flatten())[0]

        z1 = np.array((self.rot_y(joint_angles[0])[0:3, 0:3] * self.rot_z(joint_angles[1])[0:3, 0:3] * np.array([0, 0, 1]).reshape(3,1)).flatten())[0]

        z2 = np.array((self.rot_y(joint_angles[0])[0:3, 0:3] * self.rot_z(joint_angles[1])[0:3, 0:3] * self.rot_z(joint_angles[2])[0:3, 0:3] * np.array([0, 0, 1]).reshape(3,1)).flatten())[0]

        z3 = np.array((self.rot_y(joint_angles[0])[0:3, 0:3] * self.rot_z(joint_angles[1])[0:3, 0:3] * self.rot_z(joint_angles[2])[0:3, 0:3] * self.rot_y(joint_angles[3])[0:3, 0:3] * np.array([0, -1, 0]).reshape(3, 1)).flatten())[0]


        t4 = np.dot(z3, np.cross(p44, grav)[0, 0:3])

        t3 = np.dot(z2, np.cross(p33, grav)[0, 0:3]) + np.dot(z2, np.cross(p34, grav)[0, 0:3])

        t2 = np.dot(z1, np.cross(p22, grav)[0, 0:3]) + np.dot(z1, np.cross(p23, grav)[0, 0:3]) + np.dot(z1, np.cross(p24, grav)[0, 0:3])

        t1 = np.dot(z0, np.cross(p11, grav)[0, 0:3]) + np.dot(z0, np.cross(p12, grav)[0, 0:3]) + np.dot(z0, np.cross(p13, grav)[0, 0:3]) + np.dot(z0, np.cross(p14, grav)[0, 0:3])

        return np.matrix([t1, t2, t3, t4]).T

    #------NEW CODE----------------

    def phi(self, xs, ee_pos):
        sumofForce = np.matrix(np.array([0.0, 0.0, 0.0]).reshape(3, 1))
        for x in xs:
            diffX = x[0:3] - ee_pos
            dist = math.sqrt(diffX[0]*diffX[0]+diffX[1]*diffX[1]+diffX[2]*diffX[2])
            gradX = diffX/dist
            r = x[3]*4
            if dist <= r:
                sumofForce += np.matrix(np.array(((1/dist)- (1/r))*(1/(dist*dist)) * gradX).reshape(3, 1))*20
            else:
                sumofForce += np.matrix(np.array([0.0, 0.0, 0.0]).reshape(3, 1))
        return sumofForce

    #------END NEW CODE----------------

    def go(self):
        #The robot has several simulated modes:
        #These modes are listed in the following format:
        #Identifier (control mode) : Description : Input structure into step function

        #POS : A joint space position control mode that allows you to set the desired joint angles and will position the robot to these angles : env.step((np.zeros(3),np.zeros(3), desired joint angles, np.zeros(3)))
        #POS-IMG : Same control as POS, however you must provide the current joint angles and velocities : env.step((estimated joint angles, estimated joint velocities, desired joint angles, np.zeros(3)))
        #VEL : A joint space velocity control, the inputs require the joint angle error and joint velocities : env.step((joint angle error (velocity), estimated joint velocities, np.zeros(3), np.zeros(3)))
        #TORQUE : Provides direct access to the torque control on the robot : env.step((np.zeros(3),np.zeros(3),np.zeros(3),desired joint torques))
        self.env.controlMode="TORQUE"
        self.charge_obstacle = False
        #Run 100000 iterations
        prev_JAs = np.zeros(4)
        prev_jvs = np.zeros(4)

        self.red_temp = np.zeros((52,52))
        self.red_temp = cv2.circle(self.red_temp,(26,26),25,1,-1).astype(np.uint8)
        self.green_temp = np.zeros((46,46))
        self.green_temp = cv2.circle(self.green_temp,(23,23),22,1,-1).astype(np.uint8)
        self.blue_temp = np.zeros((38,38))
        self.blue_temp = cv2.circle(self.blue_temp,(19,19),18,1,-1).astype(np.uint8)
        self.end_temp = np.zeros((28,28))
        self.end_temp = cv2.circle(self.end_temp,(14,14),13,1,-1).astype(np.uint8)
        self.obst_temp = np.zeros((42,42))
        self.obst_temp = cv2.circle(self.end_temp,(41,41),40,1,-1).astype(np.uint8)

        self.obsts=[]

        prevEePos = np.zeros(shape=(1,3))


        # Uncomment to have gravity act in the z-axis
        self.env.world.setGravity((0,0,-9.81))
        #self.env.enable_gravity(True)
        self.prev_target=(0,0,0)

        for i in range(1000000):
            dt = self.env.dt
            arrxy, arrxz = self.env.render('rgb-array')

            if i == 0:
                detectedJointAngles = np.zeros(4)
                detectedJointVels = np.zeros(4)
                self.ee_target = [0, 0, 0]
            else:
                detectedJointAngles = self.detect_joint_angles(arrxy, arrxz, prev_JAs, prev_jvs)
                detectedJointVels = self.angle_normalize(detectedJointAngles-prev_JAs)/dt
                if not(self.charge_obstacle):
                    self.ee_target = self.get_target_coords(arrxy, arrxz)

            prev_jvs = detectedJointVels
            prev_JAs = detectedJointAngles

            ee_pos = np.array(self.FK(detectedJointAngles)[0:3, 3]).flatten()

            # -------------NEW CODE--------------------

            if i==1:
                self.obsts = self.detect_obstacles(arrxy,arrxz)
                if self.charge_obstacle:
                    closest_dist = float('inf')
                    for obst in self.obsts:
                        diffX = obst[0:3] - ee_pos
                        dist = np.sqrt(diffX[0]*diffX[0]+diffX[1]*diffX[1]+diffX[2]*diffX[2])
                        if dist < closest_dist and abs(obst[0])<=2 and abs(obst[0])<=2 and abs(obst[1])<=2 and abs(obst[2])<=2:
                            self.ee_target = obst[0:3]
                            closest_dist = dist

            # -------------END NEW CODE--------------------

            ee_vel = (ee_pos - prevEePos)/dt

            prevEePos = ee_pos

            J = self.Jacobian(detectedJointAngles)[0:3,:]

            ee_desired_force = self.ts_pd_control(ee_pos, ee_vel, self.ee_target)

            grav_opposite_torques = self.grav(detectedJointAngles)

            # New codes, adds phi to torque
            torques = J.T*ee_desired_force - grav_opposite_torques - J.T * self.phi(self.obsts, ee_pos)

            self.env.step((np.zeros(3),np.zeros(3), np.zeros(3), torques))

def main():
    reach = MainReacher()
    reach.go()

if __name__ == "__main__":
    main()
