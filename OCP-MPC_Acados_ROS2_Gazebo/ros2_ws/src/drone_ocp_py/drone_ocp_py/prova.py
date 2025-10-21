import numpy as np
import casadi as ca
from common import *
import math

#Ta_rot = 0.5      #0.5
#Ta_rot_yaw = 5*Ta_rot       # 5
#zita_rot = 0.7
#zita_rot_yaw = 0.7
#wn_rot = 4/(zita_rot*Ta_rot)
#wn_rot_yaw = 4/(zita_rot_yaw*Ta_rot_yaw)
#
#
#Ta_z= 10*Ta_rot #5
#Ta_xy = 10*Ta_rot #5
#zita = 0.7
#zita_xy = 0.7
#wn_z = 4/ (zita*Ta_z)
#wn_xy = 4/ (zita_xy*Ta_xy)
#
#print("wn yaw: ",wn_rot_yaw)
#print ("wn rollpitch: ", wn_rot)
#print ("wn xy: ",wn_xy)
#print("wn z: ", wn_z)
#
#
#f_yaw = wn_rot_yaw * 2*math.pi
#f_rollpitch = wn_rot * 2 * math.pi
#f_xy = wn_xy * 2 * math.pi
#f_z = wn_z * 2 * math.pi
#
#print ("f_yaw: ", f_yaw)
#print ("f_rollpitch: ",f_rollpitch)
#print ("f_xy: ", f_xy)
#print ("f_z: ", f_z)
#
#R = RPY_to_R(0.4,1,0.2)
#R = np.matrix(R)
#print(R.shape)

vx = 3
vy = 2
vz = 1

qw=0.5
qx = 0.2
qy = 0.5
qz = 0.6

q = [qw,qx,qy,qz]

R = quat_to_R(q).full()

[vx,vy,vz] = R @ [vx,vy,vz]

print(vx,vy,vz)