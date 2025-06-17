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

    # Inputs (generalized forces) in body frame
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
    Rb = quat_to_R(q)
    #zB=ca.mtimes(R,ca.vertcat(0,0,1))

    # Equations of motion (ODEs)
    p_dot = v
    v_dot = (1/m) * ca.mtimes(Rb, F) - g
    q_dot = 0.5 * mtimes(omega_matrix(w), q)    #propagazione del quaternione
    w_dot = mtimes(ca.inv(J), (tau - cross(w, mtimes(J, w))))

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
    
    ref_sym = ca.SX.sym('p', 6)  # simbolico per ref (p,rpy)
    model.p = ref_sym       #model.p = parameters 

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

#Drone model rpy
def convert_to_rpy_model(model_quat):

    # Model parameters
    g = vertcat(0,0,g0)   # gravity [m/s^2]


    # Nuove variabili di stato
    p = SX.sym('p', 3)
    v = SX.sym('v', 3)
    rpy = SX.sym('rpy', 3)
    omega = SX.sym('omega', 3)
    x = vertcat(p, v, rpy, omega)

    # Controlli
    u = model_quat.u
    F = u[:3]
    tau = u[3:]

    # Rotazione da RPY
    phi = rpy[0]
    theta=rpy[1]
    psi=rpy[2]
    Rb=RPY_to_R(phi,theta,psi)

    dp = v
    dv = (1/m) * mtimes(Rb ,F) - g

    # Derivata degli angoli di eulero
    #T = SX(3,3)
    #T[0,:] = ca.horzcat(1, sin(phi)*tan(theta), cos(phi)*tan(theta))
    #T[1,:] = ca.horzcat(0, cos(phi),           -sin(phi))
    #T[2,:] = ca.horzcat(0, sin(phi)/cos(theta), cos(phi)/cos(theta))
    #drpy = T @ omega
    drpy = angularVel_to_EulerRates(phi,theta,psi,omega)

    domega = mtimes(ca.inv(J), (tau - cross(omega, mtimes(J, omega))))

    xdot = vertcat(dp, dv, drpy, domega)

    model_rpy = type('', (), {})()
    model_rpy.x = x
    model_rpy.u = u
    model_rpy.xdot = xdot
    model_rpy.f_expl_expr = xdot
    model_rpy.name = model_quat.name + "_rpy"
    ref_sym = ca.SX.sym('p', 6)  # simbolico per ref (p,rpy)
    model_rpy.p = ref_sym       #model.p = parameters 
    model_rpy.m = m
    model_rpy.g = g
    model_rpy.J = J
        
    #define in x_labels the roll, pitch and yaw
    model_rpy.x_labels = [
        r'$x$', r'$y$', r'$z$',
        r'$v_x$', r'$v_y$', r'$v_z$',
        r'$\phi$', r'$\theta$', r'$\psi$',
        r'$\omega_x$', r'$\omega_y$', r'$\omega_z$'
    ]
    model_rpy.u_labels = [r'$F_x$', r'$F_y$', r'$F_z$', r'$\tau_x$', r'$\tau_y$', r'$\tau_z$']
    model_rpy.t_label = '$t$ [s]'

    return model_rpy
