from acados_template import AcadosOcp, AcadosOcpSolver
from drone_model import *
from common import *
from planner import *
from scipy.linalg import solve_continuous_are
import numpy as np
import casadi as ca


#################  AGGIUSTARE: ricavare snap, jerk, acc in qualche modo perché da y_expr non si può tramite get(...)
##############  Estendere lo stato con tutti gli stati


##############  Assegnare all'oggetto la traiettoria desiderata e mettere nei vincoli lo stare ad una certa distanza
##############  Includere vincoli visuali

def setup_model():
    model = export_quadrotor_ode_model()
    model_rpy = convert_to_rpy_model(model)
    return model, model_rpy

def setup_initial_conditions() :
    xx=0
    y=0
    z=0
    
    vx=0
    vy=0
    vz=0

    roll=0
    pitch=0
    yaw=0
    q = RPY_to_quat(roll,pitch,yaw)

    wx=0
    wy=0
    wz=0

    x0 = np.array([xx,y,z,vx,vy,vz,*q,wx,wy,wz])
    x0_rpy=np.array([xx,y,z,vx,vy,vz,roll,pitch,yaw,wx,wy,wz])
    return x0,x0_rpy


def configure_ocp(model, x0, p_obj, rpy_obj, Tf, ts, W, W_e, radius=2.0):
    #model dimensions
    # model.x = [p(3), v(3), quat(4), omega(3)]
    # u = [f(3), tau(3)]
    nx = model.x.rows()
    nu = model.u.rows()

    #prediction horizon time
    N_horiz = int(Tf/ts)

    #Optimization Control Problem creation
    ocp = AcadosOcp()

    #model definition: it's handled as set of equality constraints
    ocp.model = model
    
    #time: total simulation time and prediction horizon
    ocp.solver_options.tf = Tf
    ocp.solver_options.N_horizon = N_horiz

    ##########                          CONSTRAINTS             ################

    #initial conditions for constraints
    ocp.constraints.x0 = x0
    # State physical constraints
    ocp.constraints.lbx = np.array([0] + [-2]*3 + [-np.deg2rad(60)]*3)
    ocp.constraints.ubx = np.array([12] + [2]*3 + [np.deg2rad(60)]*3)
    ocp.constraints.idxbx = np.array([2, 3, 4, 5, 10, 11, 12])
    # Control constraints
    Fmax = 20  #more or less double that hovering
    Tmax = 1
    ocp.constraints.lbu = np.array([-Fmax, -Fmax, -Fmax, -Tmax, -Tmax, -Tmax])
    ocp.constraints.ubu = np.array([Fmax, Fmax, Fmax, Tmax, Tmax, Tmax])
    ocp.constraints.idxbu = np.arange(nu)
    # Solver options
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
    #ocp.solver_options.hessian_approx = 'EXACT'

    ##########                          COST FUNCTION               ##############

    #Converting quaternion to rpy
    rpy_expr = quat_to_RPY(model.x[6:10])

    # ========== Dynamics extraction ========== #
    xdot = model.f_expl_expr  # explicit model

    #Position
    p_expr = model.x[0:3]

    # Velocity
    v_expr = model.x[3:6]  # v = dot(p)

    # Angular velocity + Euler rates
    w_expr = model.x[10:]
    dot_rpy = angularVel_to_EulerRates(rpy_expr[0],rpy_expr[1],rpy_expr[2],w_expr)

    # Acceleration (is part of xdot)
    a_expr = xdot[3:6]

############################################################################################################                   
    #Jerk
    j_expr = ca.jacobian(a_expr, model.x) @ xdot                
                                                                            #APPROSSIMAZIONE DERIVATE (non tenendo conto
                                                                            # di u e u_dot da cui j e s dipendono)                                           
    # Snap = symbolic time derivative of jerk (d/dt(j)= ...)                # valutare se espandere lo stato                               
    s_expr = ca.jacobian(j_expr, model.x) @ xdot             
############################################################################################################
    u_hovering = ca.DM([0, 0, m*g0, 0, 0, 0])
    
    # Substitution of u with u_hovering to obtain acc, jerk, snap "at hovering" for last time instant (no dependance on model.u)
    a_hover = ca.substitute(a_expr, model.u, u_hovering)
    j_hover = ca.substitute(j_expr, model.u, u_hovering)
    s_hover = ca.substitute(s_expr, model.u, u_hovering)

    ############## REFERENCES
    #importing references as sym from model, for y_expr    
    #POSITION
    p_ref_sym=model.p[0:3]
    p_rel_sym = p_expr - p_ref_sym
    r_expr = ca.norm_2(p_rel_sym)                       #radius
    pan_expr = ca.arctan2(p_rel_sym[0], p_rel_sym[1])   #polar angle
    tilt_expr = ca.arcsin(p_rel_sym[2]/r_expr)          #azimuth angle
    dist_drone_obj= p_expr[0]-p_ref_sym[0]

    #ORIENTATION
    rpy_ref_sym=model.p[3:]
    R_obj = RPY_to_R(rpy_ref_sym[0],rpy_ref_sym[1],rpy_ref_sym[2])
    R_drone = RPY_to_R(rpy_expr[0], rpy_expr[1], rpy_expr[2])
    mutual_R = R_obj.T * R_drone
    mutual_rot = R_to_RPY(mutual_R)
    
    # Const function quantities (expressed with respect to state and control)
    y_expr = ca.vertcat(
        r_expr,
        pan_expr,
        tilt_expr,
        v_expr,                         # velocity
        *mutual_rot,
        dot_rpy,                        # Euler rates
        a_expr,                         # acceleration
        j_expr,                         # jerk
        s_expr,                         # snap
        model.u                         # control
    )
    # Last time instant expression (senza model.u)
    y_expr_e = ca.vertcat(
        r_expr,
        pan_expr,
        tilt_expr,
        v_expr,                         # velocity
        *mutual_rot,
        dot_rpy,                        # Euler rates
        a_hover,                        # acceleration
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
    # initial values of parametes (p_obj,rpy_obj) (they will be updated at each iteration )
    ocp.parameter_values = np.concatenate([p_obj[0,:],rpy_obj[0,:]])  

    #############       REFERENCES
    
    # Before task
    r_ref = radius
    pan_ref = 0.0
    tilt_ref = 0.0
    mutual_rot_ref = np.array([0.0, 0.0, 0.0])
    dot_rpy_ref = np.array([0,0,0])
    v_ref=np.array([0,0,0])
    acc_ref=np.array([0,0,0])
    jerk_ref=np.array([0,0,0])
    snap_ref=np.array([0,0,0])
    u_ref=np.zeros(nu)

    final_mut_pos = np.array([radius+2, 0, pi/4])   # r pan e tilt
    rpy_final_mut_rot = np.array([pi/6, 0, 0])          ###### FARE IN MODO CHE SIA ORIENTATO COME JOYSTICK

    #dist_pos_ind = slice(0,3)
    r_ind = 0
    pan_ind = 1
    tilt_ind = 2
    vel_ind = slice(tilt_ind+1,tilt_ind + 4)
    rpy_ind = slice(vel_ind.stop, vel_ind.stop+3)
    dot_rpy_ind = slice(rpy_ind.stop,rpy_ind.stop+3)
    acc_ind = slice(dot_rpy_ind.stop,dot_rpy_ind.stop+3)
    jerk_ind = slice(acc_ind.stop,acc_ind.stop+3)
    snap_ind = slice(jerk_ind.stop,jerk_ind.stop+3)
    u_ind = slice(snap_ind.stop,snap_ind.stop+6)
    
    # initialization of cost references for state and input 
    # over prediction horizon and last time istant
    yref = np.zeros(y_expr.numel())
    yref_e = np.zeros(y_expr_e.numel())

    # ASSIGN REFERENCES
    yref[r_ind]= r_ref   #distance 
    yref[pan_ind] = pan_ref
    yref[tilt_ind] = tilt_ref
    yref[vel_ind]=v_ref
    yref[rpy_ind]=mutual_rot_ref
    yref[dot_rpy_ind]=dot_rpy_ref
    yref[acc_ind]=acc_ref
    yref[jerk_ind]=jerk_ref
    yref[snap_ind]=snap_ref
    yref[u_ind]=u_ref

    #for last tract of trajectoy (task)
    new_ref = np.concatenate([final_mut_pos, v_ref, rpy_final_mut_rot, yref[dot_rpy_ind.start:]])

    #Terminal reference
    yref_e = new_ref[:yref_e.shape[0]]

    ocp.cost.yref = yref
    ocp.cost.yref_e = yref_e

    #solver creation
    ocp.solver_options.nlp_solver_max_iter=300
    ocp_solver = AcadosOcpSolver(ocp)


    # Definition of cost references (ocp_solver.yref) and substituting pos and ang reference values from sym to real
    for i in range(N_horiz):
        param = np.concatenate([p_obj[i],rpy_obj[i]])
        ocp_solver.set(i,"p",param)

        if (i>0.7*N_horiz):
            #param = np.concatenate([final_mut_pos,rpy_final_mut_rot])
            #ocp_solver.set(i,"p",param)
            ocp_solver.set(i,"yref", new_ref)
    ocp_solver.set(N_horiz,"p",param)
    ocp_solver.set(N_horiz,"yref", new_ref[:yref_e.shape[0]])

    #x_ref_final = np.concatenate([p_ref_i, v_ref_i, rpy_ref_i, dot_rpy_ref_i ,a_ref_i,j_ref_i, s_ref_i])
    #x_ref_final = np.array(p_ref)

    return ocp_solver, N_horiz, nx, nu

def extract_trajectory_from_solver(ocp_solver, model, N_horiz, nx, nu):

    y_expr=model.cost_y_expr

    fun_y = ca.Function('fun_y', [model.x, model.u,model.p], [y_expr])
    simX = np.array([ocp_solver.get(i, "x") for i in range(N_horiz + 1)])
    simU = np.array([ocp_solver.get(i, "u") for i in range(N_horiz)])
    simP = np.array([ocp_solver.get(i, "p") for i in range(N_horiz + 1)])

        # Valuto y_expr per ogni istante (tranne l'ultimo che non ha controllo u)
    y_vals = []
    for i in range(N_horiz):
        y_i = fun_y(simX[i], simU[i],simP[i])
        y_vals.append(y_i.full().flatten())
    # Per l'ultimo step (senza controllo), uso u=0 (oppure u dell'ultimo step)
    y_last = fun_y(simX[-1], simU[-1], simP[-1])
    y_vals.append(y_last.full().flatten())

    y_vals = np.array(y_vals)  # (N_horiz+1) x dimensione_y_expr

    # Ora estrai gli indici relativi ad accelerazione, jerk, snap all'interno di y_expr
    #               y_expr structure:
    # y_expr = vertcat(
    #    model.x[0:6],        # p(0:6), v(3:6)
    #    rpy_expr,            # RPY  (3)
    #    model.x[10:13],      # omega (3)
    #    a_expr,              # acceleration (3)
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

    return simX, simU, a, j, s

def get_state_variables(simX):
    p = simX[:, 0:3]
    v = simX[:, 3:6]
    q = simX[:, 6:10]
    rpy = np.array([quat_to_RPY(qi).T for qi in q])
    rpy=np.squeeze(rpy)
    w = simX[:, 10:13]
    return p, v, rpy, w

#def drone_ref_from_obj(p_obj,rpy_obj,radius) :
    # Reference for the drone based on distance from the object
    p_obj = []
    rpy_obj = []
    for poss, rott in zip(p_obj, rpy_obj):
        R_obj = RPY_to_R(*rott)
        offset_dir = R_obj[:, 0]
        p_drone = poss + radius * offset_dir
        p_obj.append(p_drone)
        #vec_to_obj = p_obj - p_drone
        roll = 0
        pitch = 0
        yaw = 0
        rpy_obj.append([float(roll), float(pitch), float(yaw)])
    p_obj = np.squeeze(np.array(p_obj))
    rpy_obj = np.squeeze(np.array(rpy_obj))
    return p_obj, rpy_obj 

