from acados_template import AcadosOcp, AcadosOcpSolver
from drone_ocp_py.drone_model import *
from drone_ocp_py.common import *
from drone_ocp_py.planner import *
#from scipy.linalg import solve_continuous_are
import numpy as np
import casadi as ca
from scipy.spatial.transform import Rotation 

#################  AGGIUSTARE: ricavare snap, jerk, acc in qualche modo perché da y_expr non si può tramite get(...)
##############  Estendere lo stato con tutti gli stati

def build_yref_online(y_idx, ref_vec):
    yref = np.zeros(y_idx["u"].stop) 
    
    # ref_vec ora contiene [X_target, Y_target, Z_target, qw, qx, qy, qz]
    yref[y_idx["pos"]]     = ref_vec[0:3]          # Posizione Cartesiana X, Y, Z
    yref[y_idx["vel"]]     = np.array([0,0,0])
    yref[y_idx["quat"]]    = ref_vec[3:7]          # Quaternione puro w, x, y, z
    yref[y_idx["dot_rpy"]] = np.array([0,0,0])
    yref[y_idx["acc"]]     = np.array([0,0,0])
    yref[y_idx["acc_ang"]] = np.array([0,0,0])
    yref[y_idx["jerk"]]    = np.array([0,0,0])
    yref[y_idx["snap"]]    = np.array([0,0,0])
    yref[y_idx["u"]]       = np.zeros(4)
    return yref

def build_yref_terminal(y_idx, ref_vec, ny_e):
    y = build_yref_online(y_idx, ref_vec)
    return y[:ny_e]  


def setup_model(m, Ixx, Iyy, Izz):
    model = export_quadrotor_ode_model(m, Ixx, Iyy, Izz)
    model_rpy = convert_to_rpy_model(model, m, Ixx, Iyy, Izz)
    return model, model_rpy

def setup_initial_conditions(start_x,start_y,start_z,start_phi,start_theta,start_psi) :
    xx = start_x
    y =  start_y
    z =  start_z
    
    vx  = 0
    vy  = 0
    vz  = 0

    roll =  start_phi
    pitch = start_theta
    yaw =   start_psi

    q=Rotation.from_euler('xyz', [roll, pitch, yaw]).as_quat()
    qw,qx,qy,qz = np.roll(q,1)

    wx=0
    wy=0
    wz=0

    x0 = np.array([xx,y,z,vx,vy,vz,qw,qx,qy,qz,wx,wy,wz])
    x0_rpy=np.array([xx,y,z,vx,vy,vz,roll,pitch,yaw,wx,wy,wz])
    return x0,x0_rpy

def set_initial_state(ocp_solver, xk):
    ocp_solver.set(0, "lbx", xk)
    ocp_solver.set(0, "ubx", xk)

def configure_mpc(model, x0, camera_offset, p_obj, rpy_obj, Tf, ts, W, W_e, ref = np.zeros(7), final_ref = np.zeros(7)):
    
    nx = model.x.rows()
    nu = model.u.rows()

    m=model.m
    N_horiz = int(Tf/ts)

    ocp = AcadosOcp()
    ocp.model = model
    
    ocp.solver_options.tf = Tf
    ocp.solver_options.N_horizon = N_horiz

    '''
                                            STATE & KINEMATICS             
    '''
    # Position - Cartesiana Pura
    p_expr = model.x[0:3]

    # Quaternione di stato - Orientamento Puro
    q_expr = model.x[6:10]
    
    # Derivata per Euler rates (necessario per il costo sulle velocità angolari)
    rpy_expr = quat_to_RPY(q_expr)
    w_expr = model.x[10:]
    dot_rpy = angularVel_to_EulerRates(rpy_expr[0],rpy_expr[1],rpy_expr[2],w_expr)

    # Rotazione attuale del drone rispetto al world
    R_expr = quat_to_R(q_expr)

    # --- Parte visuale --> Sistema camera ---  
    d_cam = ca.DM(camera_offset[0:3]).reshape((3,1))
    p_cam_expr = p_expr + R_expr @ d_cam   # Posizione della camera nel mondo

    fov_h_rad = 80.0 * ca.pi / 180.0
    fov_v_rad = 60.0 * ca.pi / 180.0

    T_h = ca.tan(fov_h_rad / 2.0)
    T_v = ca.tan(fov_v_rad / 2.0)

    p_obj_expr = model.p[0:3]
    p_rel_world = p_obj_expr - p_cam_expr

    P_c = R_expr.T @ p_rel_world    # Posa relativa dell'oggetto rispetto alla camera, nella terna camera

    X_c = P_c[0]
    Y_c = P_c[1]
    Z_c = P_c[2]

    # state dynamics vector
    xdot = model.f_expl_expr  


    # Velocity
    v_expr = model.x[3:6]  

    # Acceleration 
    acc_expr = xdot[3:6]
    acc_ang_expr = xdot[-3:]  

    #########################################################################################################                   
    #Jerk
    j_expr = ca.jacobian(acc_expr, model.x) @ xdot                
    #j_expr= ca.SX.zeros(3,1)
    #                                                                                          
    # Snap 
    s_expr = ca.jacobian(j_expr, model.x) @ xdot
    #s_expr= ca.SX.zeros(3,1)             
    #########################################################################################################
    
    u_hovering = ca.DM([m*g0, 0, 0, 0])
    acc_hover = ca.substitute(acc_expr, model.u, u_hovering)
    acc_ang_hover = ca.substitute(acc_ang_expr, model.u, u_hovering)
    j_hover = ca.substitute(j_expr, model.u, u_hovering)
    s_hover = ca.substitute(s_expr, model.u, u_hovering)



    '''
                                            CONSTRAINTS             
    '''
    ocp.constraints.x0 = x0
    ocp.constraints.lbx = np.array([0] + [-np.pi/3]*3)  # zmin, wmin  
    ocp.constraints.ubx = np.array([100] + [np.pi/3]*3)  # zmax, wmax
    ocp.constraints.idxbx = np.array([2,-3, -2, -1])   

    Fmax = 4*9.8*m  
    Tmax = [0.25, 0.25, 0.15]
    ocp.constraints.lbu = np.array([0, -Tmax[0], -Tmax[1], -Tmax[2]])
    ocp.constraints.ubu = np.array([Fmax, Tmax[0], Tmax[1], Tmax[2]])
    ocp.constraints.idxbu = np.arange(nu)

    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    #ocp.solver_options.qp_solver_cond_N = 5 # Scommentare per abilitare un condensing parziale per velocizzare ulteriormente
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING'

    ################## PROVARE A INCLUDERE I VINCOLI 1 PER VOLTA

    # ==========================================================
    # Constraints for camera
    # ==========================================================
    visual_constr_expr = ca.vertcat(
        Y_c - T_h * X_c,  # Limite Destro:  Y_c <= T_h * X_c
        Y_c + T_h * X_c,  # Limite Sinistro: Y_c >= -T_h * X_c
        Z_c - T_v * X_c,  # Limite Alto:    Z_c <= T_v * X_c
        Z_c + T_v * X_c,  # Limite Basso:   Z_c >= -T_v * X_c
        X_c               # Profondità:     X_c >= X_min
    )
    
    model.con_h_expr = visual_constr_expr
    
    X_min = 0.5 # Il peg deve stare almeno a X_min DAVANTI alla telecamera (distanza di sicurezza)
    ocp.constraints.lh = np.array([-100,  0.0, -100,  0.0, X_min])
    ocp.constraints.uh = np.array([ 0.0,  100,  0.0,  100, 100])

    # ==========================================================
    # SOFT CONSTRAINTS (Slack Variables)
    # ==========================================================
    n_soft_h = 5
    ocp.constraints.idxsh = np.array(range(n_soft_h))

    # Usare valori strettamente positivi
    penalty_L1 = 1e0
    penalty_L2 = 1e1
    weights_costs = np.array([1, 1, 1, 1, 1])

    ocp.cost.Zl = penalty_L2 * weights_costs
    ocp.cost.Zu = penalty_L2 * weights_costs
    ocp.cost.zl = penalty_L1 * weights_costs
    ocp.cost.zu = penalty_L1 * weights_costs

  #  # --- Fine parte visuale --- 

############ GESTIRE RPY VS CENTRO IMMAGINE
######## POI GESTIRE INPUT UMANO IN MODO CHE IL DRONE TENGA INQUADRATO L'OGGETTO ANCHE CON CAMBIO RIFERIMENTO DI POSIZIONE UMANO
    '''
                                        COST FUNCTION               
    '''
    # --- COST EXPRESSION ---
    # Cost function quantities (expressed with respect to state and control)
    y_expr = ca.vertcat(
        p_cam_expr,                     # Posizione attuale della camera (X,Y,Z)
        v_expr,                         # velocity
        q_expr,                         # Orientamento attuale (w,x,y,z)
        dot_rpy,                        # Euler rates
        acc_expr,                       # acceleration
        acc_ang_expr,                   # angular acceleration
        j_expr,                         # jerk
        s_expr,                         # snap
        model.u                         # control
    )
    
    # Terminal cost exrpession
    y_expr_e = ca.vertcat(
        p_cam_expr,                     # Posizione attuale (X,Y,Z)
        v_expr,                         # velocity
        q_expr,                         # Orientamento attuale (w,x,y,z)
        dot_rpy,                        # Euler rates
        acc_hover,                      # acceleration
        acc_ang_hover,
        j_hover,                        # jerk
        s_hover,                        # snap
    )
    
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ocp.model.cost_y_expr = y_expr
    ocp.model.cost_y_expr_e = y_expr_e
    
    ocp.cost.W = W
    ocp.cost.W_e = W_e
    ocp.cost.set = True
    
    # I parametri ora passati al modello (p) sono solo informativi per chi chiama get, non servono più a y_expr
    ocp.parameter_values = np.concatenate([p_obj[0,:],rpy_obj[0,:], np.zeros(3)])  

    '''
                                        REFERENCES
    '''
    
    # Definition of constant references for derivatives
    dot_rpy_ref = np.array([0,0,0])
    v_ref=np.array([0,0,0])
    acc_ref=np.array([0,0,0])
    acc_ang_ref = np.array([0,0,0])
    jerk_ref=np.array([0,0,0])
    snap_ref=np.array([0,0,0])
    u_ref=np.zeros(nu)


    # Indexes (Aggiornati per le nuove dimensioni: pos=3, quat=4)
    pos_ind = slice(0,3)
    vel_ind = slice(pos_ind.stop,pos_ind.stop+3)
    #rpy_ind = slice(vel_ind.stop, vel_ind.stop+4)
    quat_ind = slice(vel_ind.stop, vel_ind.stop+4)
    dot_rpy_ind = slice(quat_ind.stop,quat_ind.stop+3)
    acc_ind = slice(dot_rpy_ind.stop,dot_rpy_ind.stop+3)
    acc_ang_ind = slice(acc_ind.stop,acc_ind.stop+3)
    jerk_ind = slice(acc_ang_ind.stop,acc_ang_ind.stop+3)   
    snap_ind = slice(jerk_ind.stop,jerk_ind.stop+3)
    u_ind = slice(snap_ind.stop,snap_ind.stop+4)

    y_idx = {
        "pos": pos_ind,
        "vel": vel_ind,
        "quat": quat_ind,
        "dot_rpy": dot_rpy_ind,
        "acc": acc_ind,
        "acc_ang": acc_ang_ind,
        "jerk": jerk_ind,
        "snap": snap_ind,
        "u": u_ind,
    }
    ny   = y_idx["u"].stop   
    ny_e = y_idx["u"].start  
    
    yref = np.zeros(y_expr.numel())
    yref_e = np.zeros(y_expr_e.numel())

    # ASSIGN REFERENCES (Dummy initial references, they will be overwritten online)
    yref[pos_ind]= ref[0:3]         # Target assoluto X, Y, Z
    yref[vel_ind]=v_ref             
    yref[quat_ind]= ref[3:7]        # Target assoluto w,x,y,z
    yref[dot_rpy_ind]=dot_rpy_ref   
    yref[acc_ind]=acc_ref           
    yref[acc_ang_ind]=acc_ang_ref
    yref[jerk_ind]=jerk_ref         
    yref[snap_ind]=snap_ref         
    yref[u_ind]=u_ref               

    new_ref = yref.copy()
    new_ref[pos_ind]=final_ref[0:3]
    new_ref[quat_ind]=final_ref[3:7]

    yref_e = new_ref[:y_expr_e.numel()]  

    ocp.cost.yref = yref
    ocp.cost.yref_e = yref_e

    ocp.solver_options.nlp_solver_max_iter=200
    ocp_solver = AcadosOcpSolver(ocp)

    return ocp_solver, N_horiz, nx, nu, y_idx, ny, ny_e

# Le funzioni extract_trajectory_from_solver e get_state_variables rimangono invariate per non rompere eventuale codice di plotting esterno, 
# sebbene idx_j_start ecc andrebbero riadattati alla nuova lunghezza di y_expr per estrarre correttamente jerk e snap nei log.