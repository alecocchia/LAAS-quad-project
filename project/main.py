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
    p_obj_in = np.array([1, 1 , 1])    
    p_obj_f =  np.array([10, 10, 10])
    rot_obj_in =np.array([0,0,0]) 
    rot_obj_f =np.array([0,0,0]) 
    ref_in  = np.concatenate([p_obj_in,rot_obj_in])
    ref_f   = np.concatenate([p_obj_f,rot_obj_f])

    # Reference trajectory (Object) (orientation in RPY)
    traj_time, p_obj, rpy_obj = generate_trapezoidal_trajectory(ref_in, ref_f, t0, Tf, ts, v_max=1.0, a_max=1.0)

    # Drone reference
    radius = 2
    #p_refs, rpy_refs = drone_ref_from_obj(p_obj,rpy_obj,radius)

    # Cost weights
    W_x = np.diag([1, 1, 1,        #r, pan, tilt
                    0.7, 0.7, 0.7,    #v
                    0.8, 0.8, 0.8,   #mutual_rot
                    0.5, 0.5, 0.5])  #euler_rates
    W_a = np.diag([0.8]*3)        #accel
    W_j = np.diag([0.5]*3)        #jerk
    W_s = np.diag([0.5]*3)        #snap
    W_u = np.diag([0.1]*6)    #control


    #W_e = ca.diagcat(W_x,W_a,W_j,W_s).full()
    #W = ca.diagcat(W_x,W_a,W_j,W_s, W_u).full()
    W   = ca.diagcat(W_x, W_a, W_j, W_s, W_u).full()
    W_e = ca.diagcat(W_x,W_a,W_j,W_s).full()    

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
    err_pos_norm = np.zeros((N_horiz+1,1))
    err_or_norm = np.zeros((N_horiz+1,1))

    #get norm of v,a,j,s and of error
    for i in range (N_horiz + 1) :
        vel_norm[i]  = np.linalg.norm(v[i])
        acc_norm[i]  = np.linalg.norm(acc[i])
        jerk_norm[i] = np.linalg.norm(jerk[i])
        snap_norm[i] = np.linalg.norm(snap[i])
        err_pos_norm[i]  = np.linalg.norm(p_obj[i]-p[i])
        err_or_norm[i]  =   np.linalg.norm(rpy_obj[i,0]-rpy[i,0])   #roll


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
    radius_array=np.repeat(radius,len(traj_time)).reshape(-1,1)

    # Animated Plot
    traj_plot3D_animated(traj_time,p_obj, p, labels=['Ref object', 'Drone trajectory'], colors=['blue', 'red'])
    #Error norm - plot
    myPlotWithReference(traj_time, radius_array, err_pos_norm, other_labels[0],"Distance of drone from object", 2)
    myPlotWithReference(traj_time, np.zeros(len(traj_time)).reshape(-1,1), err_or_norm, other_labels[1],"Roll distance from ref value", 2)

    #Velocity, acceleration, jerk, snap norms - plot
    myPlot(traj_time,np.hstack([vel_norm, acc_norm, jerk_norm, snap_norm]),other_labels[2:], "Norms of velocity, acceleration, jerk and snap",2)
    #States - plot
    myPlotWithReference(traj_time, p_obj, x_rpy, model_rpy.x_labels, "States", 2)
    #Control input - plot
    myPlot(traj_time[0:-1], simU, model_rpy.u_labels, "Control laws", 2)
    


if __name__ == "__main__":
    main()