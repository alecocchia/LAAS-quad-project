import numpy as np
import casadi as ca
from common import *
angoloRPY = np.array([ca.pi/4,0,ca.pi/2])
print("Angolo RPY: ",np.rad2deg(angoloRPY))
R = RPY_to_R(*angoloRPY)
print("Rot mat da RPY: ", R)
angoloRPY2 = R_to_RPY(R)
print("Angolo RPY da Rot mat: ",np.rad2deg(angoloRPY2))

print(angle_min(-ca.pi))