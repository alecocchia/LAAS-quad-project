#                MAIN
from drone_ocp_settings import *
from casadi import pi as pi

############################## CAPIRE PERCHÉ È UN PROBLEMA IL RIFERIMENTO NEGATIVO

def main():
    # Time
    t0 = 0.0
    Tf = 10.0
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
    traj_time, p_obj, rpy_obj = generate_trapezoidal_trajectory(ref_obj_in, ref_obj_f, t0, Tf, ts, v_max=1.0, a_max=1.0)
 

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
    ocp_solver, N_horiz, nx, nu = configure_ocp(model, x0, p_obj, rpy_obj, Tf, ts, W, W_e, ref, final_ref)

    status = ocp_solver.solve()
    ocp_solver.print_statistics()
    if status != 0:
        raise RuntimeError(f"Acados solver failed with status {status}")
    else :
        print(ocp_solver.get_stats('time_tot'))        # total solution time (sec)


    '''
                                            SIMULATION RESULTS
    '''
    # Simulation results extraction (simP: pos_obj, or_obj, mut_rot_references)
    simX, simU, simP, acc, jerk, snap = extract_trajectory_from_solver(ocp_solver, model, N_horiz, nx, nu)
    # Simulation results saving, orientation in RPY
    p, v, rpy, w = get_state_variables(simX)
    x_rpy=np.hstack((p, v, rpy, w))
    
    mutual_rot_ref = simP[:,6:9]
    # Cost terms for plotting
    # initialization
    dist_norm = np.zeros((N_horiz+1,1))
    mutual_rot_rpy = np.zeros((N_horiz+1, 3))
    vel_norm    = np.zeros((N_horiz+1,1))
    acc_norm    = np.zeros((N_horiz+1,1))
    jerk_norm   = np.zeros((N_horiz+1,1))
    snap_norm   = np.zeros((N_horiz+1,1))
    err_or      = np.zeros((N_horiz+1,3))
    err_or_norm = np.zeros((N_horiz+1,1))

    # Computing terms
    for i in range (N_horiz + 1) :
        dist_norm[i]  = np.linalg.norm(p_obj[i]-p[i])
        vel_norm[i]  = np.linalg.norm(v[i])

        # orientation
        R_obj = RPY_to_R(rpy_obj[i,0],rpy_obj[i,1],rpy_obj[i,2])
        R_drone = RPY_to_R(rpy[i,0], rpy[i,1], rpy[i,2])
        mutual_R = ca.mtimes(R_drone.T, R_obj)
        mutual_rot_rpy[i] = np.squeeze(R_to_RPY(mutual_R))
        err_or[i] = (mutual_rot_ref[i]-mutual_rot_rpy[i])
        err_or[i]= [min_angle(err_or[i,j]) for j in range(3)]

        err_or_norm[i]= np.linalg.norm(err_or[i])
        acc_norm[i]  = np.linalg.norm(acc[i])
        jerk_norm[i] = np.linalg.norm(jerk[i])
        snap_norm[i] = np.linalg.norm(snap[i])

    # saving constant references as vectors
    ref_vec = np.repeat(ref.reshape(1,-1), len(traj_time), axis = 0)
    final_ref_vec = np.repeat(final_ref.reshape(1,-1), len(traj_time), axis = 0)
    # conversion of orientation in degrees and unwrapping for not having fake discontinuities
    drone_rpy_deg = np.rad2deg(np.unwrap(rpy, axis = 0))
    mutual_rot_rpy_deg = np.rad2deg(np.unwrap(mutual_rot_rpy, axis = 0))
    

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

    # Animated Plot
    traj_plot3D_animated_with_orientation(traj_time, p, rpy, p_obj, rpy_obj)

    # Error norms - plot
    print(ref_vec[:,[0]].shape)
    print(final_ref_vec[:,[0]].shape)
    print(dist_norm.shape)
    print(mutual_rot_ref.shape)

    myPlotWithReference(traj_time, [ref_vec[:,[0]], final_ref_vec[:,[0]]], dist_norm, other_labels[0],"Distance of drone from object [m]", 2)
    myPlotWithReference(traj_time, [np.rad2deg(mutual_rot_ref)], mutual_rot_rpy_deg, model_rpy.x_labels[6:9], "Mutual orientation through Euler angles [deg]", 2)
    
    myPlotWithReference(traj_time, [], np.rad2deg(err_or_norm), other_labels[1], "Error norm of mutual orientation through Euler angles [deg]", 2)


    # Position and orientations - plot
    myPlotWithReference(traj_time, [p_obj], p, model_rpy.x_labels[0:3], "Positions [m]", 2)
    myPlotWithReference(traj_time, [], drone_rpy_deg , model_rpy.x_labels[6:9], "Orientations [deg]", 2)

    # Velocity, acceleration, jerk, snap norms - plot
    myPlot(traj_time,np.hstack([vel_norm, acc_norm, jerk_norm, snap_norm]),other_labels[2:], "Norms of velocity, acceleration, jerk and snap",2)    

    # Control input - plot
    myPlot(traj_time[0:-1], simU, model_rpy.u_labels, "Control laws", 2)
    


if __name__ == "__main__":
    main()