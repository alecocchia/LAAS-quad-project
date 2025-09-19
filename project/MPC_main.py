#                MAIN
from drone_MPC_settings import *  
from casadi import pi as pi


################ RIPARTIRE DA: CAPIRE BENE BENE IL CODICE, ESPORTARE IN ROS DOVE ANDRA' GESTITO DINAMICAMENTE IL CAMBIO DI RIFERIMENTO TRAMITE TOPIC
################ CONFRONTARE PLOT CASO OCP E CASO MPC E VEDERE SE CI SONO DIFFERENZE


def build_yref_online(y_idx, ref_vec):
    yref = np.zeros(y_idx["u"].stop) # dimensione di y_expr (fino a u)
    # crea array lungo quanto y_expr
    # riempimento dei campi coerente con drone_MPC_settings
    yref[y_idx["pos"]]     = ref_vec[0:3]          # [radius, pan, tilt]
    yref[y_idx["vel"]]     = np.array([0,0,0])
    yref[y_idx["quat"]]    = np.array([1,0,0,0])   # mutual rotation error as quaternion
    yref[y_idx["dot_rpy"]] = np.array([0,0,0])
    yref[y_idx["acc"]]     = np.array([0,0,0])
    yref[y_idx["acc_ang"]] = np.array([0,0,0])
    yref[y_idx["jerk"]]    = np.array([0,0,0])
    yref[y_idx["snap"]]    = np.array([0,0,0])
    yref[y_idx["u"]]       = np.zeros(4)
    return yref

def build_yref_terminal(y_idx, ref_vec, ny_e):
    y = build_yref_online(y_idx, ref_vec)
    return y[:ny_e]  # tronca alla dimensione del costo terminale


def main():
    # Time
    t0 = 0.0
    T_tot = 10.0 # durata volo oggetto
    T_p = 1 #tempo di predizione
    ts = 0.01

    #drone model setup (both quaternion and rpy)
    model, model_rpy = setup_model()

    #initial conditions for drone
    x0,x0_rpy=setup_initial_conditions()            # tutto 0
    R0 = RPY_to_R(x0_rpy[6],x0_rpy[7],x0_rpy[8])
    zb0=R0[:,2] # z axis of body frame, initial orientation
    # Hover thrust
    hover_thrust = m * g0 # *zb0
    #u_hover = np.concatenate([hover_thrust, np.zeros(3)])

    '''
                                            OBJECT TRAJECTORY
    '''
    #OBJECT reference trajectory specification
    p_obj_in = np.array([2 , 2 , 0])    
    p_obj_f =  np.array([2, 2, 10])
    rot_obj_in =np.array([0,0,0]) 
    rot_obj_f =np.array([0,0,0]) 
    ref_obj_in  = np.concatenate([p_obj_in,rot_obj_in])
    ref_obj_f   = np.concatenate([p_obj_f,rot_obj_f])

    # Object trajectory (orientation in RPY)
    traj_time, p_obj, rpy_obj = generate_trapezoidal_trajectory(ref_obj_in, ref_obj_f, t0, T_tot, ts, v_max=1.0, a_max=1.0)
 

    '''
                                            REFERENCES
    '''
    # Mutual position and orientation references
    radius = 2
    mut_pos_ref = np.array([radius, 0.0, 0.0]) # distance, pan and tilt
    mut_rot_ref = np.array([0, 0, pi/2])     # rad
    # Task

    mut_pos_final_ref = np.array([radius, 0, 0]) # distance, pan and tilt
    mut_rot_final_ref = np.array([0, 0, pi])                # rad

    ref = np.concatenate([mut_pos_ref, mut_rot_ref])
    final_ref = np.concatenate([mut_pos_final_ref, mut_rot_final_ref])

    # Human flag for changing dynamically the trajectory
    human_flag=False


    '''
                                            WEIGHTS
    '''
    #NORMALIZE WEIGTHS
    D = 10          # m
    PANTILT = 2*pi    # rad
    V = 5           # m/s
    #ANG = 2*pi        # rad
    ANG = 1            # quat
    ANG_DOT = pi/3   # rad/s
    ACC = 6        # m/s^2
    ACC_ANG = 200 
    JERK = 20       # m/s^3
    SNAP = 200       # m/s^4
    U_F = 40        # N
    U_TAU = 0.3       # N*m    

    # Weights construction
    #Q_pos = np.diag([20 / (D**2), 20 / (PANTILT**2), 20 / (PANTILT**2)])
    #Q_vel = np.diag([1]*3)/V**2
    #Q_rot = np.diag([10, 10, 5])/ANG**2
    #Q_ang_dot = np.diag([0]*3)/ANG_DOT**2
    #Q_acc = np.diag([0.1]*3)/ACC**2
    #Q_acc_ang = np.diag([0]*3)/ACC_ANG**2
    #Q_jerk = np.diag([0.2]*3)/JERK**2
    #Q_snap = np.diag([0.2]*3)/SNAP**2

    Q_pos = np.diag([10 / (D**2), 10 / (PANTILT**2), 10 / (PANTILT**2)])
    Q_vel = np.diag([2]*3)/V**2
    Q_rot = np.diag([1, 5, 5, 5])/ANG**2
    Q_ang_dot = np.diag([3, 3, 4])/ANG_DOT**2
    Q_acc = np.diag([1]*3)/ACC**2
    Q_acc_ang = np.diag([1]*3)/ACC_ANG**2
    Q_jerk = np.diag([0.2]*3)/JERK**2
    Q_snap = np.diag([0.2]*3)/SNAP**2
    
    R_f = np.diag([0.01])/U_F**2
    R_tau = np.diag([0.1]*3)/U_TAU**2
    R = ca.diagcat(R_f,R_tau)
    Q = ca.diagcat(Q_pos, Q_vel, Q_rot, Q_ang_dot, Q_acc, Q_acc_ang, Q_jerk, Q_snap)

    # Cost weights to be passed to solver
    W   = ca.diagcat(Q,R).full()
    W_e = 10 * Q.full() 

    '''
                                            SOLVER
    '''
    # configuring and solving OCP
    ocp_solver, N_horiz, nx, nu, y_idx, ny, ny_e = configure_ocp(model, x0, p_obj, rpy_obj, T_p, ts, W, W_e, ref, final_ref)

    # --- MPC loop ---
    N_sim = int((T_tot-t0)/ts)                   # simulazione lunga come la traiettoria
    t = t0
    xk = x0.copy()

    # warm-start buffers
    u_prev = [np.zeros(nu) for _ in range(N_horiz)]
    x_prev = [x0.copy() for _ in range(N_horiz+1)]

    # per logging
    X_log       = np.empty((N_sim+1, nx))   # stati
    U_log       = np.empty((N_sim,   nu))   # ingressi
    Param0_log  = np.empty((N_sim+1,     9))  # [p_obj(3), rpy_obj(3), mut_rot_des(3)]
    Yref0_log   = np.empty((N_sim+1,    ny))  # yref del nodo 0 (se ti serve)

    # k = 0 (prima del loop)
    X_log[0] = x0.copy()
    Param0_log[0] = np.concatenate([p_obj[0], rpy_obj[0], mut_rot_ref])
    Yref0_log[0]  = build_yref_online(y_idx, np.concatenate([mut_pos_ref, mut_rot_ref]))

    for k in range(N_sim):
        # 1) Stato corrente
        set_initial_state(ocp_solver, xk)   # vincolo x(0)=xk
        if k < N_sim*0.7:
            online_ref = np.concatenate([mut_pos_ref, mut_rot_ref])
        else:
            online_ref = np.concatenate([mut_pos_final_ref, mut_rot_final_ref])

        # 2) Aggiorna parametri & yref lungo la finestra [t, t+Tf]
        for i in range(N_horiz+1):
            ti = t + i*ts
            # parametri: [p_obj(t), rpy_obj(t), mut_rot_ref(t)]
            # qui usi i profili precomputati (discreti) p_obj, rpy_obj
            # prendi l'indice più vicino:
            idx = min(int((ti - t0)/ts), len(traj_time)-1)
            p_i   = p_obj[idx]
            rpy_i = rpy_obj[idx]
            param = np.concatenate([p_i, rpy_i, online_ref[3:]])
            ocp_solver.set(i, "p", param)
            if i < N_horiz:
                yref_i = build_yref_online(y_idx, online_ref)
                ocp_solver.set(i, "yref", yref_i)
            if i == 0:
                param0 = param.copy()  # [p_obj(0:3), rpy_obj(3:6), mut_rot_des(6:9)]
                yref0  = yref_i.copy() if i < N_horiz else None

        # terminal yref (stessa logica, ma senza input)
        yref_e = build_yref_online(y_idx, online_ref)[:ny_e]
        ocp_solver.set(N_horiz, "yref", yref_e)

        # 3) warm-start (opzionale)
        for i in range(N_horiz):
            ocp_solver.set(i, "u", u_prev[i])
            ocp_solver.set(i, "x", x_prev[i])
        ocp_solver.set(N_horiz, "x", x_prev[N_horiz])

        # 4) solve
        status = ocp_solver.solve()
        if status != 0:
            raise RuntimeError(f"Acados status {status}")
        else :
            print(ocp_solver.get_stats('time_tot'))

        # 5) Applica u0 e “avanza” lo stato usando la predizione
        u0 = ocp_solver.get(0, "u")
        x_next = ocp_solver.get(1, "x")   # salva next x predetto

        # 6) Warm-start shift
        for i in range(N_horiz-1):
            u_prev[i] = ocp_solver.get(i+1, "u")
            x_prev[i] = ocp_solver.get(i+1, "x")
        u_prev[N_horiz-1] = u_prev[N_horiz-2].copy() if N_horiz > 1 else ocp_solver.get(0,"u").copy()
        x_prev[N_horiz]   = ocp_solver.get(N_horiz, "x")

        # salvataggi
        U_log[k]      = np.asarray(u0).ravel()
        X_log[k+1]    = np.asarray(x_next).ravel()
        Param0_log[k+1] = np.asarray(param0).ravel()     # quello che hai passato a i=0
        Yref0_log[k+1]  = np.asarray(yref0).ravel()      # opzionale
        
        # avanzamento
        xk = X_log[k+1].copy()
        t += ts

    t_x = t0 + np.arange(N_sim+1)*ts
    t_u = t0 + np.arange(N_sim)*ts

    # Stati (RPY) dal log
    p, v, rpy, w = get_state_variables(X_log)
    x_rpy = np.hstack((p, v, rpy, w))

    # Riferimenti usati ad ogni iterazione (nodo 0)
    p_obj_hist   = Param0_log[:, 0:3]                # (N_sim, 3)
    rpy_obj_hist = Param0_log[:, 3:6]                # (N_sim, 3)
    mut_rot_ref_hist = Param0_log[:, 6:9]            # (N_sim, 3)

    # Allinea lunghezze: p, v, rpy sono lunghezza N_sim (come i log)
    # (se hai una differenza di +1, tronca o interpoli a tua scelta)
    L = len(t_x)
    p   = p[:L]; v = v[:L]; rpy = rpy[:L]; w = w[:L]
    # --- riferimento posizione assoluto per il plot ---
    # estraggo dal log cosa ho usato davvero ad ogni k
    p_obj_hist        = Param0_log[:, 0:3]         # (N,3)  posizione oggetto
    rpy_obj_hist      = Param0_log[:, 3:6]         # (N,3)  orientazione oggetto
    mut_rel_sph       = Yref0_log[:, y_idx["pos"]] # (N,3)  [r, pan, tilt] usati nel nodo 0

    # sferiche -> cartesiane nel frame OGGETTO
    r   = mut_rel_sph[:, 0]
    pan = mut_rel_sph[:, 1]
    tilt= mut_rel_sph[:, 2]
    rel_obj = np.column_stack([
        r*np.cos(tilt)*np.cos(pan),
        r*np.cos(tilt)*np.sin(pan),
        r*np.sin(tilt)
    ])

    # ruota nel WORLD con R_obj(k) e somma p_obj(k)
    R_obj_stack = np.array([RPY_to_R(*angles).full() for angles in rpy_obj_hist])  # (N,3,3)
    rel_world   = np.einsum('nij,nj->ni', R_obj_stack, rel_obj)                    # (N,3)
    p_ref_world = p_obj_hist + rel_world                                           # (N,3)
    
    # Errori e derivate
    dist_norm = np.linalg.norm(p - p_obj_hist, axis=1, keepdims=True)
    vel_norm  = np.linalg.norm(v,            axis=1, keepdims=True)

    # Calcolo grezzo di acc/jerk/snap via differenze finite
    acc       = np.diff(v, axis=0, prepend=v[[0], :]) / ts
    jerk      = np.diff(acc, axis=0, prepend=acc[[0], :]) / ts
    snap      = np.diff(jerk, axis=0, prepend=jerk[[0], :]) / ts
    acc_norm  = np.linalg.norm(acc,  axis=1, keepdims=True)
    jerk_norm = np.linalg.norm(jerk, axis=1, keepdims=True)
    snap_norm = np.linalg.norm(snap, axis=1, keepdims=True)

    # Rotazione reciproca effettiva e errore
    mutual_rot_rpy = np.zeros_like(rpy)
    for i in range(L):
        R_obj   = RPY_to_R(rpy_obj_hist[i,0], rpy_obj_hist[i,1], rpy_obj_hist[i,2])
        R_drone = RPY_to_R(rpy[i,0],          rpy[i,1],          rpy[i,2])
        mutual_R = ca.mtimes(R_drone.T, R_obj)
        mutual_rot_rpy[i,:] = np.squeeze(R_to_RPY(mutual_R))

    err_or = (mut_rot_ref_hist - mutual_rot_rpy)
    for j in range(3):
        err_or[:, j] = [min_angle(val) for val in err_or[:, j]]
    err_or_norm = np.linalg.norm(err_or, axis=1, keepdims=True)

    # vettori di riferimento "a gradino" (se vuoi visualizzare i target nominali)
    ref_vec       = np.repeat(ref.reshape(1, -1),       L, axis=0)
    final_ref_vec = np.repeat(final_ref.reshape(1, -1), L, axis=0)

    # conversione in gradi e unwrap per i plot
    drone_rpy_deg      = np.rad2deg(np.unwrap(rpy,            axis=0))
    mutual_rot_rpy_deg = np.rad2deg(np.unwrap(mutual_rot_rpy, axis=0))
    

    '''
                                            PLOTTING
    '''    

    # Plot drone states and controls
    # plot_drone(t_sim, 20, simU, x, True, True, model_rpy.t_label, model_rpy.x_labels, model_rpy.u_labels)
    other_labels = [
        rf"|| p_d(t)-p_o(t) ||",
        rf"|| mutRot_des(t)- mut_Rot(t)||",
        rf"||v(t)||",
        rf"||a(t)||",
        rf"||j(t)||",
        rf"||s(t)||"
        ]

    # Animazione 3D
    traj_plot3D_animated_with_orientation(t_x, p, rpy, p_obj_hist, rpy_obj_hist)

    # Errori
    myPlotWithReference(t_x, [ref_vec[:, [0]], final_ref_vec[:, [0]]],
                        dist_norm, r"|| p_d(t)-p_o(t) ||", "Distance of drone from object [m]", 2)

    myPlotWithReference(t_x, [np.rad2deg(mut_rot_ref_hist)],
                        mutual_rot_rpy_deg, model_rpy.x_labels[6:9], "Mutual orientation through Euler angles [deg]", 2)

    myPlotWithReference(t_x, [], np.rad2deg(err_or_norm),
                        r"|| mutRot_des(t) - mut_Rot(t) ||", "Error norm of mutual orientation through Euler angles [deg]", 2)

    # Stati
    myPlotWithReference(t_x, [p_ref_world], p, model_rpy.x_labels[0:3], "Positions [m]", 2)
    myPlotWithReference(t_x, [], drone_rpy_deg, model_rpy.x_labels[6:9], "Orientations [deg]", 2)

    # Norme vel/acc/jerk/snap
    myPlot(t_x, np.hstack([vel_norm, acc_norm, jerk_norm, snap_norm]),
           [rf"||v(t)||", rf"||a(t)||", rf"||j(t)||", rf"||s(t)||"],
           "Norms of velocity, acceleration, jerk and snap", 2)

    # Ingressi di controllo (usi il log MPC)
    U_log = np.vstack(U_log)                     # (L, nu)
    myPlot(t_u, U_log, model_rpy.u_labels, "Control laws", 2)



if __name__ == "__main__":
    main()