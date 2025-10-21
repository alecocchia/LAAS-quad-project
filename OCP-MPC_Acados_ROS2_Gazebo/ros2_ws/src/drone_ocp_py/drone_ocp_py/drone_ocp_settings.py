from acados_template import AcadosOcp, AcadosOcpSolver
from drone_ocp_py.drone_model import *
from drone_ocp_py.common import *
from drone_ocp_py.planner import *
from scipy.linalg import solve_continuous_are
import numpy as np
import casadi as ca
from scipy.spatial.transform import Rotation 

#################  AGGIUSTARE: ricavare snap, jerk, acc in qualche modo perché da y_expr non si può tramite get(...)
##############  Estendere lo stato con tutti gli stati


##############  Assegnare all'oggetto la traiettoria desiderata e mettere nei vincoli lo stare ad una certa distanza
##############  Includere vincoli visuali

def setup_model():
    model = export_quadrotor_ode_model()
    model_rpy = convert_to_rpy_model(model)
    return model, model_rpy

def setup_initial_conditions() :
    xx = 0
    y =  0
    z =  0
    
    vx  = 0
    vy  = 0
    vz  = 0

    roll =  0
    pitch = 0
    yaw =   0

    q=Rotation.from_euler('xyz', [roll, pitch, yaw]).as_quat()
    qw,qx,qy,qz = np.roll(q,1)

    wx=0
    wy=0
    wz=0

    x0 = np.array([xx,y,z,vx,vy,vz,qw,qx,qy,qz,wx,wy,wz])
    x0_rpy=np.array([xx,y,z,vx,vy,vz,roll,pitch,yaw,wx,wy,wz])
    return x0,x0_rpy


def configure_ocp(model, x0, p_obj, rpy_obj, Tf, ts, W, W_e, ref = np.zeros(6), final_ref = np.zeros(6)):
    
    # model:
    # model.x = [p(3), v(3), quat(4), omega(3)]
    # u = [f(3), tau(3)]
    # Dimensions
    nx = model.x.rows()
    nu = model.u.rows()


    #prediction horizon time
    N_horiz = int(Tf/ts)

    # creation of Optimization Control Problem
    ocp = AcadosOcp()

    # model definition: it's handled as set of equality constraints
    ocp.model = model
    
    # time: total simulation time and prediction horizon
    ocp.solver_options.tf = Tf
    ocp.solver_options.N_horizon = N_horiz
    ocp.solver_options

    '''
                                            CONSTRAINTS             
    '''

    # initial conditions for constraints
    ocp.constraints.x0 = x0
    # State physical constraints
    ocp.constraints.lbx = np.array([0] + [np.deg2rad(-60)]*3)  # zmin, wmin  
    ocp.constraints.ubx = np.array([100] + [np.deg2rad(60)]*3)  # zmax, wmax
    #ocp.constraints.lbx =  np.array([0])      # zmin
    #ocp.constraints.ubx =  np.array([100] )   # zmax
    ocp.constraints.idxbx = np.array([2,-3, -2, -1])   # constrained variables indexes
    # Control constraints
    Fmax = 4*m*g0  #more or less 4 times than hovering
    Tmax = [0.25, 0.25, 0.15]
    ocp.constraints.lbu = np.array([0, -Tmax[0], -Tmax[1], -Tmax[2]])
    ocp.constraints.ubu = np.array([Fmax, Tmax[0], Tmax[1], Tmax[2]])
    ocp.constraints.idxbu = np.arange(nu)
    # Solver options
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
    #ocp.solver_options.hessian_approx = 'EXACT'


    '''
                                        COST FUNCTION               
    '''

    # ========== Dynamics extraction ========== #
    xdot = model.f_expl_expr  # explicit model

    #Position
    p_expr = model.x[0:3]

    #Converting quaternion to rpy
    rpy_expr = quat_to_RPY(model.x[6:10])

    # Velocity
    v_expr = model.x[3:6]  # v = dot(p)

    # Angular velocity + Euler rates
    w_expr = model.x[10:]
    dot_rpy = angularVel_to_EulerRates(rpy_expr[0],rpy_expr[1],rpy_expr[2],w_expr)

    # Acceleration (is part of xdot)
    acc_expr = xdot[3:6]
    acc_ang_expr = xdot[-3:]  #per ora per semplicità è la derivata di w (omega)
############################################################################################################                   
    #Jerk
    j_expr = ca.jacobian(acc_expr, model.x) @ xdot                
                                                                            #APPROSSIMAZIONE DERIVATE (non tenendo conto
                                                                            # di u e u_dot da cui j e s dipendono)                                           
    # Snap = symbolic time derivative of jerk (d/dt(j)= ...)                # valutare se espandere lo stato                               
    s_expr = ca.jacobian(j_expr, model.x) @ xdot             
############################################################################################################
    u_hovering = ca.DM([m*g0, 0, 0, 0])
    
    # Substitution of u with u_hovering to obtain acc, jerk, snap "at hovering" for last time instant (no dependance on model.u)
    acc_hover = ca.substitute(acc_expr, model.u, u_hovering)
    acc_ang_hover = ca.substitute(acc_ang_expr, model.u, u_hovering)
    j_hover = ca.substitute(j_expr, model.u, u_hovering)
    s_hover = ca.substitute(s_expr, model.u, u_hovering)

    '''
                                    POSITION AND ORIENTATION COSTS DEFINITION
    '''

    #importing references as sym from model, for y_expr    
    #POSITION
    p_obj_sym = model.p[0:3]
    p_rel_sym = p_expr - p_obj_sym

    r_expr = ca.norm_2(p_rel_sym)                       #radius
    pan_expr = ca.arctan2(p_rel_sym[1], p_rel_sym[0])   #polar angle
    tilt_expr = ca.arcsin(p_rel_sym[2]/r_expr)          #azimuth angle
    
    p_rel_expr = np.array([r_expr, pan_expr, tilt_expr])    #relative position in spherical coordinates

    #ORIENTATION
    rpy_obj_sym = model.p[3:6]
    mut_rot_ref = model.p[6:9]

    R_obj = RPY_to_R(rpy_obj_sym[0],rpy_obj_sym[1],rpy_obj_sym[2])
    R_drone = RPY_to_R(rpy_expr[0], rpy_expr[1], rpy_expr[2])
    mutual_R = ca.mtimes(R_drone.T, R_obj)

    mutual_R_ref = RPY_to_R(mut_rot_ref[0],mut_rot_ref[1],mut_rot_ref[2])
    mutual_R_error = ca.mtimes(mutual_R.T, mutual_R_ref)

    #mutual_rot_error = [min_angle(R_to_RPY(mutual_R_error))]
    #mutual_rot_rpy = R_to_RPY(mutual_R)
    #mutual_rot_error = [min_angle(mut_rot_ref-mutual_rot_rpy)]
    mutual_rot_error = R_to_quat(mutual_R_error)

    
    '''
                                        COST EXPRESSIONS
    '''


    # Cost function quantities (expressed with respect to state and control)
    y_expr = ca.vertcat(
        *p_rel_expr,
        v_expr,                         # velocity
        mutual_rot_error,
        dot_rpy,                        # Euler rates
        acc_expr,                       # acceleration
        acc_ang_expr,                     # angular acceleration
        j_expr,                         # jerk
        s_expr,                         # snap
        model.u                         # control
    )
    # Last time instant expression (senza model.u)
    y_expr_e = ca.vertcat(
        *p_rel_expr,
        v_expr,                         # velocity
        mutual_rot_error,
        dot_rpy,                        # Euler rates
        acc_hover,                      # acceleration
        acc_ang_hover,
        j_hover,                        # jerk
        s_hover,                        # snap
    )
    
    # type of cost funtion
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    # expression to weight in cost function
    ocp.model.cost_y_expr = y_expr
    ocp.model.cost_y_expr_e = y_expr_e
    # Weigths over prediction horizon and last time istant
    ocp.cost.W = W
    ocp.cost.W_e = W_e
    ocp.cost.set = True
    # initial values of parametes (p_obj,rpy_obj) (they will be updated at each iteration )
    ocp.parameter_values = np.concatenate([p_obj[0,:],rpy_obj[0,:], ref[3:6]])  

    '''
                                        REFERENCES
    '''
    
    # Before task
    mut_pos_ref = ref[0:3]
    mut_rot_ref = ref[3:6]
    dot_rpy_ref = np.array([0,0,0])
    v_ref=np.array([0,0,0])
    acc_ref=np.array([0,0,0])
    acc_ang_ref = np.array([0,0,0])
    jerk_ref=np.array([0,0,0])
    snap_ref=np.array([0,0,0])
    u_ref=np.zeros(nu)

    # Task 
    final_mut_pos_ref = final_ref[0:3]   # r pan e tilt
    final_mut_rot_ref = final_ref[3:6]          ###### FARE IN MODO CHE SIA ORIENTATO COME JOYSTICK
    #for i in range(3,6) : 
    #    if np.abs(min_angle(final_ref[i]-ref[i])) < np.abs(final_ref[i]):
    #        final_mut_rot_ref[i-3] =  ref[i] + min_angle(final_ref[i]-ref[i])

    # Indexes
    pos_ind = slice(0,3)
    vel_ind = slice(pos_ind.stop,pos_ind.stop+3)
    quat_ind = slice(vel_ind.stop, vel_ind.stop+4)
    dot_rpy_ind = slice(quat_ind.stop,quat_ind.stop+3)
    acc_ind = slice(dot_rpy_ind.stop,dot_rpy_ind.stop+3)
    acc_ang_ind = slice(acc_ind.stop,acc_ind.stop+3)
    jerk_ind = slice(acc_ang_ind.stop,acc_ang_ind.stop+3)   # cambiare questo se non si include acc_ang
    snap_ind = slice(jerk_ind.stop,jerk_ind.stop+3)
    u_ind = slice(snap_ind.stop,snap_ind.stop+4)
    
    # initialization of cost references for state and input 
    # over prediction horizon and last time istant
    yref = np.zeros(y_expr.numel())
    yref_e = np.zeros(y_expr_e.numel())

    # ASSIGN REFERENCES
    yref[pos_ind]= mut_pos_ref   # distance, pan and tilt
    yref[vel_ind]=v_ref             # velocity
    yref[quat_ind]= [1.0,0.0,0.0,0.0]          # mutual rotation error rpy
    yref[dot_rpy_ind]=dot_rpy_ref   # Euler rates
    yref[acc_ind]=acc_ref           # acceleration
    yref[acc_ang_ind]=acc_ang_ref
    yref[jerk_ind]=jerk_ref         # jerk
    yref[snap_ind]=snap_ref         # snap
    yref[u_ind]=u_ref               # control

    #for last tract of trajectoy (task)
    new_ref = yref.copy()
    new_ref[pos_ind]=final_mut_pos_ref
    new_ref[quat_ind]=[1.0,0.0,0.0,0.0]

    #Terminal reference
    yref_e = new_ref[:y_expr_e.numel()]   #p,rpy

    ocp.cost.yref = yref
    ocp.cost.yref_e = yref_e

    #solver creation
    ocp.solver_options.nlp_solver_max_iter=200
    ocp_solver = AcadosOcpSolver(ocp)

    '''
                                    REFERENCES AND PARAMETERS 
                                        ONLINE UPDATE
    '''

    # Definition of cost references (ocp_solver.yref) and substituting pos and ang object values from sym to real
    # together with mutual_rotation_reference
    for i in range(N_horiz):
        #BEFORE TASK 
        param = np.concatenate([p_obj[i],rpy_obj[i],mut_rot_ref])
        #TASK -> change of reference
        if (i>0.7*N_horiz):
            param = np.concatenate([p_obj[i],rpy_obj[i],final_mut_rot_ref])
            ocp_solver.set(i,"yref", new_ref)
        ocp_solver.set(i,"p",param)

    param = np.concatenate([p_obj[N_horiz], rpy_obj[N_horiz], final_mut_rot_ref])
    ocp_solver.set(N_horiz,"p",param)
    ocp_solver.set(N_horiz,"yref", new_ref[:yref_e.shape[0]])

    return ocp_solver, N_horiz, nx, nu

def extract_trajectory_from_solver(ocp_solver, model, N_horiz, nx, nu):

    y_expr=model.cost_y_expr

    fun_y = ca.Function('fun_y', [model.x, model.u,model.p], [y_expr])
    simX = np.array([ocp_solver.get(i, "x") for i in range(N_horiz + 1)])
    simU = np.array([ocp_solver.get(i, "u") for i in range(N_horiz)])
    simP = np.array([ocp_solver.get(i, "p") for i in range(N_horiz + 1)])

    # Evaluate y_expr for each time instant (except for u that is until N-1)
    y_vals = []
    for i in range(N_horiz):
        y_i = fun_y(simX[i], simU[i],simP[i])
        y_vals.append(y_i.full().flatten())
    # Per l'ultimo step (senza controllo), uso u=0 (oppure u dell'ultimo step)
    y_last = fun_y(simX[-1], simU[-1], simP[-1])
    y_vals.append(y_last.full().flatten())

    y_vals = np.array(y_vals)  # (N_horiz+1) x dimensione_y_expr

    # Extraction of indexes relative to acc, jerk and snap in y_expr
    #               y_expr structure:
    # y_expr = vertcat(
    #    mut_pos,             # p_obj - p_drone (3)
    #    model.x[3:6]         # v (3)
    #    mut_rot_rpy,         # R_to_RPY (R_obj* R_drone^T)  (3)
    #    model.x[10:13],      # omega (3)
    #    acc_expr,              # acceleration (3)
    #    j_expr,              # jerk (3)
    #    s_expr,              # snap (3)
    #    model.u              # control (6)
    # )
    # Quindi gli indici sono:
    idx_a_start = 3 + 3 + 3 + 3  # p(3) + v(3) + rpy(3) + omega(3) = 12
    idx_a_end = idx_a_start + 3  # 12-15
    idx_j_start = idx_a_end  # 15
    idx_j_end = idx_j_start + 3  # 18
    idx_s_start = idx_j_end  # 18
    idx_s_end = idx_s_start + 3  # 21

    a = y_vals[:, idx_a_start:idx_a_end]
    j = y_vals[:, idx_j_start:idx_j_end]
    s = y_vals[:, idx_s_start:idx_s_end]

    return simX, simU, simP, a, j, s

def get_state_variables(simX):
    p = simX[:, 0:3]
    v = simX[:, 3:6]
    q = simX[:, 6:10]
    rpy = np.array([quat_to_RPY(qi).T for qi in q])
    rpy=np.squeeze(rpy)
    w = simX[:, 10:13]
    return p, v, rpy, w
