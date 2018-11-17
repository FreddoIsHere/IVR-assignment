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
    
    def coordinate_convert(self,pixels):
        #Converts pixels into metres
        return np.array([(pixels[0]-self.env.viewerSize/2)/self.env.resolution,-(pixels[1]-self.env.viewerSize/2)/self.env.resolution])
    
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
    '''
    def js_pd_control(self,  c_jas, c_jvs, d_jas):
        
        P = np.array([200,200,200,200])
        
        D = np.array([75,55,35, 20])
        
        P_error = np.matrix(d_jas-c_jas).T
        
        D_error = np.matrix(np.zeros(4)-c_jvs).T
        
        PD_error = np.diag(P)* P_error + np.diag(D)* D_error
        
        return PD_error
    '''
        
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
        
        
        '''
        print('link-centres:')
        print('for j1:')
        print(p11, p12, p13, p14)
        print('for j2:')
        print(p22, p23, p24)
        print('for j3:')
        print(p34, p33)
        print('for j4:')
        print(p44)
        '''
        
        z0 = np.array((self.rot_y(joint_angles[0])[0:3, 0:3] * np.array([0, -1, 0]).reshape(3,1)).flatten())[0]
    
        z1 = np.array((self.rot_y(joint_angles[0])[0:3, 0:3] * self.rot_z(joint_angles[1])[0:3, 0:3] * np.array([0, 0, 1]).reshape(3,1)).flatten())[0]
       
        z2 = np.array((self.rot_y(joint_angles[0])[0:3, 0:3] * self.rot_z(joint_angles[1])[0:3, 0:3] * self.rot_z(joint_angles[2])[0:3, 0:3] * np.array([0, 0, 1]).reshape(3,1)).flatten())[0]
        
        z3 = np.array((self.rot_y(joint_angles[0])[0:3, 0:3] * self.rot_z(joint_angles[1])[0:3, 0:3] * self.rot_z(joint_angles[2])[0:3, 0:3] * self.rot_y(joint_angles[3])[0:3, 0:3] * np.array([0, -1, 0]).reshape(3, 1)).flatten())[0]
               
        
        t4 = np.dot(z3, np.cross(p44, grav)[0, 0:3])
    
        t3 = np.dot(z2, np.cross(p33, grav)[0, 0:3]) + np.dot(z2, np.cross(p34, grav)[0, 0:3])
        
        t2 = np.dot(z1, np.cross(p22, grav)[0, 0:3]) + np.dot(z1, np.cross(p23, grav)[0, 0:3]) + np.dot(z1, np.cross(p24, grav)[0, 0:3])

        t1 = np.dot(z0, np.cross(p11, grav)[0, 0:3]) + np.dot(z0, np.cross(p12, grav)[0, 0:3]) + np.dot(z0, np.cross(p13, grav)[0, 0:3]) + np.dot(z0, np.cross(p14, grav)[0, 0:3])
        
        
        '''
        t1 = np.array(-grav *p14 - grav*p13 - grav*p12 - grav*p11)[0][0]
        
        t2 = np.array(-grav *p24 - grav*p23 - grav*p22)[0][0]
        
        t3 = np.array(-grav *p34 - grav*p33)[0][0]
        
        t4 = np.array(- grav*p44)[0][0]
        '''
    
        '''
        print('4th-t:', t4)
        print('z3:', z3)
        print('p44', p44)
        print('3th-t:', t3)
        print('z2:', z2)
        print('p33, p34', p33, p34)
        print('2th-t:', t2)
        print('z1:', z1)
        print('p22, p23, p24', p22, p23, p24)
        print('1th-t:', t1)
        print('z0:', z0)
        print('p11, p12, p13, p14', p11, p12, p13, p14)
        '''
        
        return np.matrix([t1, t2, t3, t4]).T

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
        prev_JAs = np.zeros(4)
        prev_jvs = collections.deque(np.zeros(4),1)
        
        prevEePos = np.zeros(shape=(1,3))
        
        #prevJointAngles = np.zeros(4)
    

        # Uncomment to have gravity act in the z-axis
        self.env.world.setGravity((0,0,-9.81))
        #self.env.enable_gravity(True)

        '''
        for i in range(100000):
            dt = self.env.dt
            arrxy,arrxz = self.env.render('rgb-array')     
         
            detectedJointAngles = self.env.ground_truth_joint_angles
            x, y, z = self.env.ground_truth_valid_target

            desired_jointAngles = self.IK(detectedJointAngles,[x, y, -z])+ detectedJointAngles
            
            detectedJointVels = self.angle_normalize(detectedJointAngles-prevJointAngles)/dt
            
            prevJointAngles = detectedJointAngles
            
            torques = self.js_pd_control(detectedJointAngles, detectedJointVels, desired_jointAngles)
            
            self.env.step((np.zeros(3),np.zeros(3),np.zeros(3),torques))
        '''
        
        for i in range(1000000):
            dt = self.env.dt
            arrxy, arrxz = self.env.render('rgb-array')
            detectedJointAngles = self.env.ground_truth_joint_angles
            prev_jvs.append(self.angle_normalize(detectedJointAngles-prev_JAs))
            prev_JAs = detectedJointAngles

            x, y, z = self.env.ground_truth_valid_target
            ee_target = [x, y, z]

            x, y, z = self.env.ground_truth_end_effector
            
            ee_pos = np.array([x, y, z])
            

            ee_vel = (ee_pos - prevEePos)/dt

            prevEePos = ee_pos

            J = self.Jacobian(detectedJointAngles)[0:3,:]

            ee_desired_force = self.ts_pd_control(ee_pos, ee_vel, ee_target)
            
            grav_opposite_torques = self.grav(detectedJointAngles)

            torques = J.T*ee_desired_force - grav_opposite_torques
            torques = - grav_opposite_torques

            ### Uncomment to compare the FK outputs ###
            # eePos = self.detect_ee(rgb_array)
            # eeFKAnalytic = self.FK_analytic(detectedJointAngles)
            # eeFK = self.FK(detectedJointAngles)
            # print "End effector position: " + str(eePos)
            # print "FK Analytic: " + str(eeFKAnalytic)
            # print "FK: " + str(eeFK) 
            #self.env.step((np.zeros(3),np.zeros(3), np.zeros(3), torques))
            if i < 600:
                self.env.step((np.zeros(3),np.zeros(3), [0.8,-0.2,-1.8,0.4], np.zeros(3)))
            else:
                print("Trying to hold against gravity now")
                self.env.controlMode="TORQUE"
                self.env.step((np.zeros(3),np.zeros(3), np.zeros(3), torques))
                
            
def main():
    reach = MainReacher()
    reach.go()

if __name__ == "__main__":
    main()