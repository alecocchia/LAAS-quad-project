#                MAIN
from drone_ocp_settings import *

 ###### RIPARTIRE DA: CAPIRE COME AFFRONTARE IL PROBLEMA DELLA BEST VIEW: VINCOLARE 
 # TUTTE LE COMPONENTI DELLA POSIZIONE?

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
    hover_thrust = m * g0 * zb0
    u_hover = np.concatenate([hover_thrust.full().flatten(), np.zeros(3)])
    
    #OBJECT reference trajectory specification
    p_obj_in = np.array([4 , 4 , 0])    
    p_obj_f =  np.array([10, 10, 10])
    rot_obj_in =np.array([0,0,0]) 
    rot_obj_f =np.array([0,0,0]) 
    ref_in  = np.concatenate([p_obj_in,rot_obj_in])
    ref_f   = np.concatenate([p_obj_f,rot_obj_f])

    # Reference trajectory (Object) (orientation in RPY)
    traj_time, p_obj, rpy_obj = generate_trapezoidal_trajectory(ref_in, ref_f, t0, Tf, ts, v_max=1.0, a_max=1.0)

    # Drone reference
    radius = 2
        
    #NORMALIZE WEIGTHS
    D = 10          # m
    PANTILT = 2*pi    # rad
    V = 5           # m/s
    ANG = 2*pi        # rad
    ANG_DOT = pi/3   # rad/s
    ACC = 6        # m/s^2
    JERK = 20       # m/s^3
    SNAP = 200       # m/s^4
    U_F = 40        # N
    U_TAU = 3       # N*m    

    Q_pos = np.diag([5 / (D**2), 5 / (PANTILT**2), 5 / (PANTILT**2)])
    Q_vel = np.diag([0.2]*3)/V**2
    Q_rot = np.diag([10,10,20])/ANG**2
    Q_ang_dot = np.diag([0.1,0.1,0.01])/ANG_DOT**2
    Q_acc = np.diag([0.1]*3)/ACC**2
    Q_jerk = np.diag([0.04]*3)/JERK**2
    Q_snap = np.diag([0.03]*3)/SNAP**2
    
    R_f = np.diag([0.01]*3)/U_F*2
    R_tau = np.diag([0.01,0.01,0.01])/U_TAU**2
    R = ca.diagcat(R_f,R_tau)
    Q = ca.diagcat(Q_pos, Q_vel, Q_rot, Q_ang_dot, Q_acc, Q_jerk, Q_snap)
    # Cost weights
    #W_x = np.diag([ 50, 100, 100,        #r, pan, tilt
    #                wmax/10, wmax/10, wmax/10,    #v
    #                wmax/5, wmax/5, wmax/5,   #mutual_rot
    #                wmax/10, wmax/10, wmax/10])  #euler_rates
    #W_a = np.diag([wmax/20]*3)/wmax        #accel
    #W_j = np.diag([wmax/25]*3)/wmax        #jerk
    #W_s = np.diag([wmax/30]*3)/wmax        #snap
    #W_u = np.diag([wmax/100]*6)/wmax    #control

    W   = diagcat(Q,R).full()
    W_e = 20 * Q.full()    

    # configuring and solving OCP
    ocp_solver, N_horiz, nx, nu = configure_ocp(model, x0, p_obj, rpy_obj, Tf, ts, W, W_e,radius)

    status = ocp_solver.solve()
    ocp_solver.print_statistics()
    if status != 0:
        raise RuntimeError(f"Acados solver failed with status {status}")
    else :
        print(ocp_solver.get_stats('time_tot'))        # total solution time (sec)

    # Simulation results extraction (orientation is in quat., since simulation is done in quat.)
    simX, simU, acc, jerk, snap = extract_trajectory_from_solver(ocp_solver, model, N_horiz, nx, nu)
    # Simulation results saving, orientation in RPY
    p, v, rpy, w = get_state_variables(simX)
    x_rpy=np.hstack((p, v, rpy, w))
    #stacking references on states in one matrix
    vel_norm    = np.zeros((N_horiz+1,1))
    acc_norm    = np.zeros((N_horiz+1,1))
    jerk_norm   = np.zeros((N_horiz+1,1))
    snap_norm   = np.zeros((N_horiz+1,1))
    dist_norm = np.zeros((N_horiz+1,1))
    #err_or_norm = np.zeros((N_horiz+1,1))

    #get norm of v,a,j,s and of error
    for i in range (N_horiz + 1) :
        vel_norm[i]  = np.linalg.norm(v[i])
        acc_norm[i]  = np.linalg.norm(acc[i])
        jerk_norm[i] = np.linalg.norm(jerk[i])
        snap_norm[i] = np.linalg.norm(snap[i])
        dist_norm[i]  = np.linalg.norm(p_obj[i]-p[i])
        #err_or_norm[i]  =   np.linalg.norm(rpy_obj[i]-rpy[i])   #roll


    # Plot drone states and controls
    #plot_drone(t_sim, 20, simU, x, True, True, model_rpy.t_label, model_rpy.x_labels, model_rpy.u_labels)
    other_labels = [
        rf"||p_d(t)-p_o(t)||",
        rf"||\phi_d(t)-\phi(t)||",
        rf"||v(t)||",
        rf"||a(t)||",
        rf"||j(t)||",
        rf"||s(t)||"
        ]
    
    #constant reference array
    dist1=np.repeat(radius,len(traj_time)).reshape(-1,1)
    dist2 = np.repeat(radius+2, len(traj_time)).reshape(-1,1)
    
    #conversion of orientation in degrees
    drone_rpy_deg = np.vstack([np.rad2deg(rpy[:,0]), np.rad2deg(rpy[:,1]), np.rad2deg(rpy[:,2])]).T
    # Animated Plot
    traj_plot3D_animated_with_orientation(traj_time,p, rpy, p_obj, rpy_obj)
    #Error norm - plot
    myPlotWithReference(traj_time, [dist1,dist2], dist_norm, other_labels[0],"Distance of drone from object", 2)
    #myPlotWithReference(traj_time, np.zeros(len(traj_time)).reshape(-1,1), err_or_norm, other_labels[1],"Roll distance from ref value", 2)

    #Velocity, acceleration, jerk, snap norms - plot
    myPlot(traj_time,np.hstack([vel_norm, acc_norm, jerk_norm, snap_norm]),other_labels[2:], "Norms of velocity, acceleration, jerk and snap",2)
    #States - plot
    myPlotWithReference(traj_time, [p_obj], p, model_rpy.x_labels[0:3], "Positions [m]", 2)
    myPlotWithReference(traj_time, [], drone_rpy_deg , model_rpy.x_labels[6:9], "Orientations [deg]", 2)

    #Control input - plot
    myPlot(traj_time[0:-1], simU, model_rpy.u_labels, "Control laws", 2)
    


if __name__ == "__main__":
    main()