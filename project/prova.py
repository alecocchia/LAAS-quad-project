import numpy as np
import casadi as ca
from common import *
import math
#angoloRPY = np.array([ca.pi/4,0,ca.pi/2])
#print("Angolo RPY: ",np.rad2deg(angoloRPY))
#R = RPY_to_R(*angoloRPY)
#print("Rot mat da RPY: ", R)
#angoloRPY2 = R_to_RPY(R)
#print("Angolo RPY da Rot mat: ",np.rad2deg(angoloRPY2))



#rpy_drone = ca.SX.sym("rpy_drone", 3)
#rpy_obj = ca.SX.sym("rpy_obj", 3)
#
#
#R_drone = RPY_to_R(rpy_drone[0], rpy_drone[1], rpy_drone[2])
#R_obj = RPY_to_R(rpy_obj[0], rpy_obj[1], rpy_obj[2])
#
#R_mut = ca.mtimes(R_drone, R_obj.T)
#rpy_mut = R_to_RPY(R_mut)
#
#rpy_mut_fun = ca.Function("rpy_mut_fun", [rpy_drone, rpy_obj], [rpy_mut])
#
#
#rpy_d = [0, 0, ca.pi/2]
#rpy_o = [0, 0, ca.pi]
#result = rpy_mut_fun(rpy_d, rpy_o)
#
#rpy_err = result.full().flatten()
#print(rpy_err)

print(min_angle(-ca.pi))
#print(min_angle(-ca.pi))


