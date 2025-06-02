#Drone model rpy
from common import *
from casadi import *

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
    f = u[:3]
    tau = u[3:]

    # Rotazione da RPY
    phi = rpy[0]
    theta=rpy[1]
    psi=rpy[2]
    R=RPY_to_R(phi,theta,psi)

    dp = v
    dv = (1/m) * R @ f - vertcat(0, 0, g0)

    # Derivata degli angoli di eulero
    T = SX(3,3)
    T[0,:] = ca.horzcat(1, sin(phi)*tan(theta), cos(phi)*tan(theta))
    T[1,:] = ca.horzcat(0, cos(phi),           -sin(phi))
    T[2,:] = ca.horzcat(0, sin(phi)/cos(theta), cos(phi)/cos(theta))
    drpy = T @ omega

    domega = mtimes(ca.inv(J), (tau + cross(omega, mtimes(J, omega))))

    xdot = vertcat(dp, dv, drpy, domega)

    model_rpy = type('', (), {})()
    model_rpy.x = x
    model_rpy.u = u
    model_rpy.xdot = xdot
    model_rpy.f_expl_expr = xdot
    model_rpy.name = model_quat.name + "_rpy"
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
