import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import ode
import math
import cv2
from pyquaternion import Quaternion as pyquat
from reacher3D import rendering
#------NEW CODE--------
import random
#------END CODE--------

class ReacherEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 5
    }

    def create_link(self,body,pos):
        body.setPosition(pos)
        M = ode.Mass()
        M.setCylinderTotal(1,1,0.1,1)
        M.translate((0.5,0,0))
        M.setParameters(M.mass,0,M.c[1],M.c[2],M.I[0][0],M.I[1][1],M.I[2][2],M.I[0][1],M.I[0][2],M.I[1][2])
        body.setMass(M)

    def create_ee(self,body,pos,collision):
        body.setPosition(pos)
        M = ode.Mass()
        M.setCylinderTotal(0.000000001, 1,0.000001,0.000001)
        body.setMass(M)
        collision.setBody(body)
    def init_rod_template(self):
        self.rod_template_1 = np.matrix(np.zeros((int(math.ceil(self.resolution)+10),int(math.ceil(self.resolution)+10))))
        self.rod_template_1[30:-20+int(math.ceil(self.resolution)),int(math.floor(((math.ceil(self.resolution)+10)/2.0)-(self.resolution*0.15))):int(math.ceil(((math.ceil(self.resolution)+10)/2.0)+(self.resolution*0.15)))]=1
        self.rod_template_1 = self.rod_template_1.T

        self.rod_template_2 = np.matrix(np.zeros((int(math.ceil(self.resolution)+10),int(math.ceil(self.resolution)+10))))
        self.rod_template_2[30:-20+int(math.ceil(self.resolution)),int(math.floor(((math.ceil(self.resolution)+10)/2.0)-(self.resolution*0.1))):int(math.ceil(((math.ceil(self.resolution)+10)/2.0)+(self.resolution*0.1)))]=1
        self.rod_template_2 = self.rod_template_2.T

        self.rod_template_3 = np.matrix(np.zeros((int(math.ceil(self.resolution)+10),int(math.ceil(self.resolution)+10))))
        self.rod_template_3[30:-20+int(math.ceil(self.resolution)),int(math.floor(((math.ceil(self.resolution)+10)/2.0)-(self.resolution*0.05))):int(math.ceil(((math.ceil(self.resolution)+10)/2.0)+(self.resolution*0.05)))]=1
        self.rod_template_3 = self.rod_template_3.T

    def __init__(self):

        #------NEW CODE--------
        self.use_new_reacher = True
        if self.use_new_reacher:
            self.static = True
            self.ob = []
            self.ob_transform = []
            self.ob_pos = []
            self.ob_rad = []
            self.ob_vel = []
            self.noObjs = 5
            self.obj_radius_max = 0.2
            if self.static:
                self.noObjs *= 2
            self.ob_pos_range = 6
        #------END NEW CODE--------

        self.dt=.005
        self.viewer = None
        self.viewerSize = 800
        self.spaceSize = 8.4
        self.resolution = self.viewerSize/self.spaceSize
        self.perspective_transform_on = False
        self.init_rod_template()
        self._seed()
        self.world = ode.World()
        self.world.setGravity((0,0,0))
        self.body1 = ode.Body(self.world)
        self.body2 = ode.Body(self.world)
        self.body3 = ode.Body(self.world)
        self.body5 = ode.Body(self.world)
        self.body4 = ode.Body(self.world)
        self.create_link(self.body1,(0.5,0,0))
        self.create_link(self.body2,(1.5,0,0))
        self.create_link(self.body3,(2.5,0,0))
        self.create_link(self.body5,(3.5,0,0))
        self.space = ode.Space()
        self.body4_col = ode.GeomSphere(self.space,radius=0.1)
        self.create_ee(self.body4,(4,0,0),self.body4_col)

        self.colours=[]

        self.dir=1.01

        # Connect body1 with the static environment
        self.j1 = ode.HingeJoint(self.world)
        self.j1.attach(self.body1, ode.environment)
        self.j1.setAnchor( (0,0,0) )
        #self.j1.setAxis( (0,0,1) )
        self.j1.setAxis( (0,-1,0) )
        self.j1.setFeedback(1)

        # Connect body2 with body1
        self.j2 = ode.HingeJoint(self.world)
        self.j2.attach(self.body1, self.body2)
        self.j2.setAnchor( (1,0,0) )
        self.j2.setAxis( (0,0,-1) )
        #self.j2.setAxis( (0,1,0) )
        self.j2.setFeedback(1)

        #connect body3 with body2
        self.j3 = ode.HingeJoint(self.world)
        self.j3.attach(self.body2, self.body3)
        self.j3.setAnchor( (2,0,0) )
        self.j3.setAxis( (0,0,-1) )
        #self.j3.setAxis( (0,1,0) )
        self.j3.setFeedback(1)

        #connect body5 with body3
        self.j5 = ode.HingeJoint(self.world)
        self.j5.attach(self.body3, self.body5)
        self.j5.setAnchor( (3,0,0) )
        #self.j5.setAxis( (0,0,-1) )
        self.j5.setAxis( (0,1,0) )
        self.j5.setFeedback(1)

        #connect end effector
        self.j4 = ode.FixedJoint(self.world)
        self.j4.attach(self.body5,self.body4)
        self.j4.setFixed()

        self.controlMode = "POS"
        self.targetPos = self.rand_target()

        self.targetTime = 0
        self.targetTime2 = 0
        self.success = 0
        self.fail = 0
        self.P_gains = np.array([1000,1000,1000,1000])
        self.D_gains = np.array([70,50,40,20])


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def enable_gravity(self,on):
        if on:
            self.world.setGravity((0,-9.81,0))
        else:
            self.world.setGravity((0,0,0))

    def pd_control(self,jas,jvs,target_jas):
        pos_gains = np.diag(self.P_gains)
        damp_gains = np.diag(self.D_gains) #np.matrix([[100,0,0],[0,60,0],[0,0,20]])
        return np.array(pos_gains*(np.matrix(angle_normalize(target_jas-jas))).T).flatten()-np.array((damp_gains*np.matrix(jvs).T).T).flatten()

    def pd_vel_control(self,jas,jvs):
        pos_gains = np.diag(self.P_gains)
        damp_gains = np.diag(self.D_gains) #np.matrix([[100,0,0],[0,60,0],[0,0,20]])
        return np.array(pos_gains*np.matrix(jas).T).flatten()-np.array((damp_gains*np.matrix(jvs).T).T).flatten()

    def rand_target(self):
        pos = (np.random.rand(3,1)-0.5)*4
        self.invalpos = (np.random.rand(3,1)-0.5)
        self.invalpos += np.sign(self.invalpos)* 0.2
        self.invalpos += pos
        self.targetGeom = ode.GeomSphere(self.space, radius=0.1)
        self.targetGeom.setPosition((pos[0],pos[1],pos[2]))
        spikes = [3,4,5,7,9]
        temp= np.random.randint(3)
        self.targetParams = [0.12,spikes[temp]]
        self.invalidParams= [0.08,1,0.10*np.random.rand()+0.4]
        self.target = rendering.make_valid_target(self.targetParams[0],self.targetParams[1])
        self.targetTrans = rendering.Transform()
        self.target.add_attr(self.targetTrans)
        self.invalid_target = rendering.make_invalid_target(self.invalidParams[0],self.invalidParams[1],self.invalidParams[2])
        self.invalidTargetTrans = rendering.Transform()
        self.invalid_target.add_attr(self.invalidTargetTrans)
        self.intargetGeom = ode.GeomSphere(self.space, radius=0.1)
        self.intargetGeom.setPosition((self.invalpos[0],self.invalpos[1],self.invalpos[2]))
        return pos

    def near_callback(self,args, geom1, geom2):
        """Callback function for the collide() method.

        This function checks if the given geoms do collide and
        creates contact joints if they do.
        """
        # Check if the objects do collide
        contacts = ode.collide(geom1, geom2)
        if(len(contacts)>0):
            if geom2 == self.targetGeom:
                self.targetTime+=self.dt
            if geom2 == self.intargetGeom:
                self.targetTime2+=self.dt
        else:
            if geom2 == self.targetGeom:
                self.targetTime=0
            if geom2 == self.intargetGeom:
                self.targetTime2=0

    def _step(self,(jas,jvs,target_ja,torques)):
        if(self.controlMode=="POS"):
            jointAngles = np.array([self.j1.getAngle(),self.j2.getAngle(),self.j3.getAngle(),self.j5.getAngle()])
            jointVelocities = np.array([self.j1.getAngleRate(), self.j2.getAngleRate(), self.j3.getAngleRate(), self.j5.getAngleRate()])
            output_torques = self.pd_control(jointAngles,jointVelocities,target_ja)
            self.j1.addTorque(output_torques[0])
            self.j2.addTorque(output_torques[1])
            self.j3.addTorque(output_torques[2])
            self.j5.addTorque(output_torques[3])
        if(self.controlMode=="POS-IMG"):
            torques = self.pd_control(jas,jvs,target_ja)
            self.j1.addTorque(torques[0])
            self.j2.addTorque(torques[1])
            self.j3.addTorque(torques[2])
            self.j5.addTorque(torques[3])
        if(self.controlMode=="VEL"):
            #self.D_gains = np.array([60,50,40])
            #self.P_gains = np.array([400,400,400])
            torques = self.pd_vel_control(jas,jvs)
            self.j1.addTorque(torques[0])
            self.j2.addTorque(torques[1])
            self.j3.addTorque(torques[2])
            self.j5.addTorque(torques[3])
        if(self.controlMode=="TORQUE"):
            self.j1.addTorque(torques[0])
            self.j2.addTorque(torques[1])
            self.j3.addTorque(torques[2])
            self.j5.addTorque(torques[3])
        self.world.step(self.dt)
        self.space.collide(self.world,self.near_callback)
        if(self.targetTime>1 or self.targetTime2>1):
            self.targetPos=self.rand_target()
            if(self.targetTime>1):
                self.success += 1
            else:
                self.fail +=1
            self.targetTime=0
            self.targetTime2=0
            print 'success', self.success
            print 'fail', self.fail

    def _reset(self):
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def update_colour(self, colour):
        redBright=colour[0]*self.viewer.lightValue
        greenBright=colour[1]*self.viewer.lightValue
        blueBright=colour[2]*self.viewer.lightValue
        # redBright=self.check_colour(redBright)
        # greenBright=self.check_colour(greenBright)
        # blueBright=self.check_colour(blueBright)
        return (redBright, greenBright, blueBright)

    def check_colour(self, colour):
        if colour>1:
            colour=1
        elif colour<0.15:
            colour=0.15
        return colour

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        x1,y1,z1 = self.body1.getPosition()
        x2,y2,z2 = self.body2.getPosition()
        x3,y3,z3 = self.body3.getPosition()
        x5,y5,z5 = self.body5.getPosition()
        if self.viewer is None:

            self.viewer = rendering.Viewer(self.viewerSize,self.viewerSize)
            self.viewer.perspective_transform_on=self.perspective_transform_on
            self.viewer.set_bounds(-self.spaceSize/2.0,self.spaceSize/2.0,-self.spaceSize/2.0,self.spaceSize/2.0)

            if self.use_new_reacher:
                if self.static:
                    for i in range(0,self.noObjs):
                        self.ob_rad.append(random.random()*self.obj_radius_max+0.05)
                        self.ob.append(rendering.make_sphere(self.ob_rad[i]))
                        self.ob[i].set_color(0, 0, 0)
                        self.ob_transform.append(rendering.Transform())
                        self.ob_pos.append(self.get_random_pos())
                        self.ob_vel.append(self.get_random_vel())
                        self.ob_transform[i].set_translation(self.ob_pos[i][0],self.ob_pos[i][1],self.ob_pos[i][2])
                        self.ob[i].add_attr(self.ob_transform[i])
                        self.colours.append([0.0,0.0,0.0])
                        self.viewer.add_geom(self.ob[i])

            rod1 = rendering.make_cuboid(1, .3)
            rod1.set_color(0.4, 0.4, 0.4)
            self.colours.append([0.4,0.4,0.4])
            self.pole_transform1 = rendering.Transform()
            self.pole_transform11 = rendering.Transform()
            rod1.add_attr(self.pole_transform1)
            rod1.add_attr(self.pole_transform11)



            rod2 = rendering.make_cuboid(1, .2)
            rod2.set_color(0.2, 0.2, 0.2)
            self.colours.append([0.2,0.2,0.2])
            self.pole_transform2 = rendering.Transform()
            self.pole_transform21 = rendering.Transform()
            rod2.add_attr(self.pole_transform2)
            rod2.add_attr(self.pole_transform21)


            rod3 = rendering.make_cuboid(1, .1)
            if self.use_new_reacher:
                rod3.set_color(0.05,0.05,0.05)
                self.colours.append([0.05,0.05,0.05])
            else:
                rod3.set_color(0,0,0)
                self.colours.append([0.0,0.0,0.0])
            self.pole_transform3 = rendering.Transform()
            self.pole_transform31 = rendering.Transform()
            rod3.add_attr(self.pole_transform3)
            rod3.add_attr(self.pole_transform31)


            rod5 = rendering.make_cuboid(1, .1)
            rod5.set_color(0.5,0.5,0.5)
            self.colours.append([0.5,0.5,0.5])
            self.pole_transform5 = rendering.Transform()
            self.pole_transform51 = rendering.Transform()
            rod5.add_attr(self.pole_transform5)
            rod5.add_attr(self.pole_transform51)

            axle1 = rendering.make_sphere(0.25)
            axle1.set_color(1,0,0)
            self.colours.append([1.0,0.0,0.0])

            self.axle_transform1 = rendering.Transform()
            axle1.add_attr(self.axle_transform1)

            self.axle_transform12 = rendering.Transform()
            axle1.add_attr(self.axle_transform12)
            axle2 = rendering.make_sphere(.22)
            axle2.set_color(0,1,0)
            self.colours.append([0.0,1.0,0.0])
            self.axle_transform2 = rendering.Transform()
            axle2.add_attr(self.axle_transform2)
            self.axle_transform22 = rendering.Transform()
            axle2.add_attr(self.axle_transform22)

            axle3 = rendering.make_sphere(.18)
            axle3.set_color(0,0,1)
            self.colours.append([0.0,0.0,1.0])
            self.axle_transform3 = rendering.Transform()
            axle3.add_attr(self.axle_transform3)
            self.axle_transform32 = rendering.Transform()
            axle3.add_attr(self.axle_transform32)


            axle5 = rendering.make_sphere(.13)
            axle5.set_color(0,0,0.5)
            self.colours.append([0.0,0.0,0.5])
            self.axle_transform5 = rendering.Transform()
            axle5.add_attr(self.axle_transform5)
            self.axle_transform52 = rendering.Transform()
            axle5.add_attr(self.axle_transform52)

            self.viewer.add_geom(rod1)
            self.viewer.add_geom(rod2)
            self.viewer.add_geom(rod3)
            self.viewer.add_geom(rod5)
            self.viewer.add_geom(axle1)
            self.viewer.add_geom(axle2)
            self.viewer.add_geom(axle3)

            self.viewer.add_geom(axle5)

        self.target.set_color(0.7,0.7,0.7)
        self.invalid_target.set_color(0.7,0.7,0.7)
        self.viewer.add_onetime(self.target)
        self.viewer.add_onetime(self.invalid_target)
        self.transform_link(self.body1, self.pole_transform11, self.pole_transform1, self.axle_transform12, self.axle_transform1)
        self.transform_link(self.body2,self.pole_transform21, self.pole_transform2,self.axle_transform22,self.axle_transform2)
        self.transform_link(self.body3,self.pole_transform31, self.pole_transform3,self.axle_transform32,self.axle_transform3)
        self.transform_link(self.body5,self.pole_transform51, self.pole_transform5,self.axle_transform52,self.axle_transform5)
        self.targetTrans.set_translation(self.targetPos[0],self.targetPos[1],self.targetPos[2])
        self.invalidTargetTrans.set_translation(self.invalpos[0],self.invalpos[1],self.invalpos[2])

        self.ground_truth_joint_angles = np.array([self.j1.getAngle(), self.j2.getAngle(),self.j3.getAngle(),self.j5.getAngle()])
        self.ground_truth_joint_velocities = np.array([self.j1.getAngleRate(), self.j2.getAngleRate(),self.j3.getAngleRate(),self.j5.getAngleRate()])
        self.ground_truth_valid_target = self.targetPos.T.flatten()
        self.ground_truth_invalid_target = self.invalpos.T.flatten()
        self.ground_truth_end_effector = self.body4.getPosition()


        #------START NEW STUFF---------------------------------------
        if self.use_new_reacher:
            if len(self.ob) == 0:
                for i in range(0,self.noObjs):
                    self.ob_rad.append(random.random()*self.obj_radius_max+0.05)
                    self.ob.append(rendering.make_sphere(self.ob_rad[i]))
                    self.ob[i].set_color(0, 0, 0)
                    self.ob_transform.append(rendering.Transform())
                    self.ob_pos.append(self.get_random_pos())
                    self.ob_vel.append(self.get_random_vel())
                    self.ob_transform[i].set_translation(self.ob_pos[i][0],self.ob_pos[i][1],self.ob_pos[i][2])
                    self.ob[i].add_attr(self.ob_transform[i])
                    if self.static:
                        self.colours.append([0.0,0.0,0.0])
                        self.viewer.add_geom(self.ob[i])
            if not(self.static):
                for i in range(0,self.noObjs):
                    self.ob_pos[i] = self.ob_pos[i] + self.ob_vel[i]
                    self.ob_transform[i].set_translation(self.ob_pos[i][0],self.ob_pos[i][1],self.ob_pos[i][2])
                    self.ob_vel[i] = self.get_new_vel(self.ob_vel[i],self.ob_pos[i])
                    self.viewer.add_onetime(self.ob[i])
            for i in range(0,len(self.ob_pos)):
                ee_pos = np.array([self.ground_truth_end_effector[0],self.ground_truth_end_effector[1],self.ground_truth_end_effector[2]])
                diffX = self.ob_pos[i] - ee_pos
                dist = np.sqrt(diffX[0]**2+diffX[1]**2+diffX[2]**2)[0]
                if dist <= self.ob_rad[i]+1:
                    print("ADDED OBJECT INDICATOR AT (%s,%s,%s)"%(self.ob_pos[i][0],self.ob_pos[i][1],self.ob_pos[i][2]))
                    objectIndi = rendering.make_sphere(self.ob_rad[i]/2)
                    objectIndi.set_color(255, 255, 255)
                    objectIndiTrans = rendering.Transform()
                    objectIndiTrans.set_translation(self.ob_pos[i][0],self.ob_pos[i][1],self.ob_pos[i][2])
                    objectIndi.add_attr(objectIndiTrans)
                    self.viewer.add_onetime(objectIndi)


        #------END NEW STUFF---------------------------------------

        if self.viewer.time==0:
            if np.random.rand()>0.5:
                self.dir=1.01
            else:
                self.dir=0.99
            self.viewer.time=np.random.randint(20)+1
        else:
            self.viewer.time-=1
        self.viewer.lightValue*=self.dir
        if self.viewer.lightValue>1:
            self.viewer.lightValue=1
        elif self.viewer.lightValue<0.01:
            self.viewer.lightValue=0.01
        for obj in self.viewer.onetime_geoms:
            redBright, greenBright, blueBright=self.update_colour(obj._color.vec4)
            obj.set_color(redBright, greenBright, blueBright)
        for i in range(np.size(self.viewer.geoms)):
            redBright, greenBright, blueBright=self.update_colour(self.colours[i])
            self.viewer.geoms[i].set_color(redBright, greenBright, blueBright)

        return self.viewer.render(True)

    #------START NEW STUFF---------------------------------------
    def get_random_pos(self):
        found_overlap = True
        while found_overlap:
            pos = (np.random.rand(3,1)-0.5)*self.ob_pos_range
            found_overlap = False
            for i in range(0,len(self.ob_pos)):
                other_pos = self.ob_pos[i]
                diffX = pos - other_pos
                dist_xy = np.sqrt(diffX[0]**2+diffX[1]**2)[0]
                dist_xz = np.sqrt(diffX[0]**2+diffX[2]**2)[0]
                if dist_xy <= self.ob_rad[i]*2+self.ob_rad[-1]*2+0.1 or dist_xz <= self.ob_rad[i]*2+self.ob_rad[-1]*2+0.1:
                    found_overlap = True
                    break
        return pos

    def get_random_vel(self):
        return (np.random.rand(3,1)-0.5)*0.05+0.01

    def get_new_vel(self, vel, pos):
        if abs(pos[0]) >= self.ob_pos_range/2.0:
            vel[0] = vel[0] * -1
        if abs(pos[1]) >= self.ob_pos_range/2.0:
            vel[1] = vel[1] * -1
        if abs(pos[2]) >= self.ob_pos_range/2.0:
            vel[2] = vel[2] * -1
        return vel

    #------END NEW STUFF---------------------------------------

    def transform_link(self,body, t1, t2, j1, j2):
        x1,y1,z1 = body.getPosition()
        (w,x,y,z) = body.getQuaternion()
        t1.set_translation(x1,y1,z1)
        t1.set_rotation(pyquat(w,x,y,z))
        t2.set_translation(-0.5,0,0)
        j1.set_translation(x1,y1,z1)
        j1.set_rotation(pyquat(w,x,y,z))
        j2.set_translation(0.5,0,0)

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
