#DRONE MODEL

from acados_template import AcadosModel
import numpy as np
from casadi import * 
from common import *

def export_quadrotor_ode_model() -> AcadosModel:

    model_name = 'quadrotor_ode'

    # Model parameters
    m = 1.0  # mass [kg]
    g = vertcat(0,0,g0)   # gravity [m/s^2]
    Ixx, Iyy, Izz = 0.015, 0.015, 0.007 #Inertia
    J = SX(np.diag([Ixx, Iyy, Izz])) #Inertia

    # States
    # Position
    px, py, pz = SX.sym('px'), SX.sym('py'), SX.sym('pz')
    p = vertcat(px, py, pz)

    # Linear velocity
    vx, vy, vz = SX.sym('vx'), SX.sym('vy'), SX.sym('vz')
    v = vertcat(vx, vy, vz)

    # Quaternion (orientation)
    qw = SX.sym('qw')
    qx = SX.sym('qx')
    qy = SX.sym('qy')
    qz = SX.sym('qz')
    q = vertcat(qw, qx, qy, qz)
    
    # Angular velocity
    wx, wy, wz = SX.sym('wx'), SX.sym('wy'), SX.sym('wz')
    w = vertcat(wx, wy, wz)

    # Inputs (generalized forces)
    Fx = SX.sym('Fx')
    Fy = SX.sym('Fy')
    Fz = SX.sym('Fz')
    tau_x = SX.sym('tau_x')
    tau_y = SX.sym('tau_y')
    tau_z = SX.sym('tau_z')
    F = vertcat(Fx, Fy, Fz)
    tau = vertcat(tau_x, tau_y, tau_z)
    u = vertcat(F, tau)

    # Rotation matrix from quaternion
    R = quat_to_R(q)
    #zB=ca.mtimes(R,ca.vertcat(0,0,1))

    # Equations of motion (ODEs)
    p_dot = v
    v_dot = (1/m) * ca.mtimes(R, F) - g
    q_dot = 0.5 * mtimes(omega_matrix(w), q)    #propagazione del quaternione
    w_dot = mtimes(ca.inv(J), (tau + cross(w, mtimes(J, w))))

    # Compose state and xdot
    x = vertcat(p, v, q, w)
    xdot=SX.sym('xdot',x.shape,)

    f_expl = vertcat(p_dot,v_dot,q_dot,w_dot)
    f_impl = xdot - f_expl

    # Define model
    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name

    #define in x_labels the roll, pitch and yaw instead of quaternion
    #so, in the plot function it is required to pass to rpy
    model.x_labels = [
        r'$x$', r'$y$', r'$z$',
        r'$v_x$', r'$v_y$', r'$v_z$',
        r'$q$', r'$q_x$', r'$q_y$', r'$q_z$',
        r'$\omega_x$', r'$\omega_y$', r'$\omega_z$'
    ]
    model.u_labels = [r'$F_x$', r'$F_y$', r'$F_z$', r'$\tau_x$', r'$\tau_y$', r'$\tau_z$']
    model.t_label = '$t$ [s]'

    return model
