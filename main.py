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

    def coordinate_convert(self,pixels):
        #Converts pixels into metres
        return np.array([(pixels[0]-self.env.viewerSize/2)/self.env.resolution,-(pixels[1]-self.env.viewerSize/2)/self.env.resolution])

    def go(self):
        #The robot has several simulated modes:
        #These modes are listed in the following format:
        #Identifier (control mode) : Description : Input structure into step function

        #POS : A joint space position control mode that allows you to set the desired joint angles and will position the robot to these angles : env.step((np.zeros(3),np.zeros(3), desired joint angles, np.zeros(3)))
        #POS-IMG : Same control as POS, however you must provide the current joint angles and velocities : env.step((estimated joint angles, estimated joint velocities, desired joint angles, np.zeros(3)))
        #VEL : A joint space velocity control, the inputs require the joint angle error and joint velocities : env.step((joint angle error (velocity), estimated joint velocities, np.zeros(3), np.zeros(3)))
        #TORQUE : Provides direct access to the torque control on the robot : env.step((np.zeros(3),np.zeros(3),np.zeros(3),desired joint torques))
        self.env.controlMode="TORQUE"
        #Run 100000 iterations
        prev_JAs = np.zeros(3)
        prev_jvs = collections.deque(np.zeros(3),1)

        # Uncomment to have gravity act in the z-axis
        # self.env.world.setGravity((0,0,-9.81))

        for _ in range(100000):
            #The change in time between iterations can be found in the self.env.dt variable
            dt = self.env.dt
            #self.env.render returns 2 RGB arrays of the robot, one for the xy-plane, and one for the xz-plane
            arrxy,arrxz = self.env.render('rgb-array')

            jointAngles = np.array([0.5,-0.5,0.5,-0.5])
            # self.env.step((np.zeros(3),np.zeros(3),jointAngles, np.zeros(3)))
            self.env.step((np.zeros(3),np.zeros(3),np.zeros(3), np.zeros(4)))
            #The step method will send the control input to the robot, the parameters are as follows: (Current Joint Angles/Error, Current Joint Velocities, Desired Joint Angles, Torque input) 

#main method
def main():
    reach = MainReacher()
    reach.go()

if __name__ == "__main__":
    main()