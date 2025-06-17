#common.py
#UTILITY FUNCTIONS
import numpy as np
import casadi as ca
import os
from pathlib import Path
from typing import Union
from scipy.spatial.transform import Rotation as R
from scipy.linalg import solve_continuous_are

from numpy.linalg import matrix_rank
from scipy.linalg import eigvals

import matplotlib.pyplot as plt
from acados_template import latexify_plot
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D



'''Global variables'''

track="trefoil_track.txt"
g0  = 9.80665       # [m.s^2] gravitational accerelation
m   = 1.            # [kg] mass
Ixx, Iyy, Izz = 0.015, 0.015, 0.007 #Inertia
J = ca.SX(np.diag([Ixx, Iyy, Izz])) #Inertia

# RPY da matrice di rotazione
def R2rpy(R_mat):
    rpy = R.from_matrix(R_mat)
    return rpy.as_euler('rpy', degrees=False)  # RPY = Roll (X), Pitch (Y), Yaw (Z)

# Da angoli RPY a matrice di rotazione
def RPY_to_R(roll, pitch, yaw):
    cr = ca.cos(roll)
    sr = ca.sin(roll)
    cp = ca.cos(pitch)
    sp = ca.sin(pitch)
    cy = ca.cos(yaw)
    sy = ca.sin(yaw)

    # Matrice di rotazione composta R = Rz(yaw) * Ry(pitch) * Rx(roll)
    R = ca.vertcat(
        ca.horzcat(cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        ca.horzcat(sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        ca.horzcat(-sp,     cp * sr,                cp * cr)
    )
    return R

#Da mat di rotazione a rpy
def R_to_RPY(R):
    """
    Estrae roll, pitch, yaw da una matrice di rotazione R (CasADi SX/MX)
    Restituisce (roll, pitch, yaw)
    """
    pitch = -ca.asin(R[2, 0])
    roll  = ca.atan2(R[2, 1], R[2, 2])
    yaw   = ca.atan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw



# Da quaternion a matrice di rotazione
def quat_to_R(q):
    #q = q / ca.norm_2(q)  # normalizza
    w, x, y, z = q[0], q[1], q[2], q[3]
    return ca.vertcat(
        ca.horzcat(1-2*(y**2+z**2), 2*(x*y - z*w), 2*(x*z + y*w)),
        ca.horzcat(2*(x*y + z*w), 1-2*(x**2+z**2), 2*(y*z - x*w)),
        ca.horzcat(2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x**2 + y**2))
    )

#Da quaternione ad RPY
def quat_to_RPY(q):
    #q = q / ca.norm_2(q)  # normalizza
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw*qx + qy*qz)
    cosr_cosp = 1 - 2 * (qx*qx + qy*qy)
    roll = ca.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (qw*qy - qz*qx)
    pitch = ca.asin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw*qz + qx*qy)
    cosy_cosp = 1 - 2 * (qy*qy + qz*qz)
    yaw = ca.atan2(siny_cosp, cosy_cosp)
    
    return ca.vertcat(roll, pitch, yaw)

#Da RPY a quaternione
def RPY_to_quat(roll, pitch, yaw):
    cr = ca.cos(roll / 2)
    sr = ca.sin(roll / 2)
    cp = ca.cos(pitch / 2)
    sp = ca.sin(pitch / 2)
    cy = ca.cos(yaw / 2)
    sy = ca.sin(yaw / 2)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    q = np.array([qw, qx, qy, qz])
    q = q / np.linalg.norm(q)  # Normalizza il quaternione

    return q

#matrice di propagazione quaternione
def omega_matrix(w):
    wx, wy, wz = w[0], w[1], w[2]
    return ca.vertcat(
        ca.horzcat(0,   -wx, -wy, -wz),
        ca.horzcat(wx,   0,  wz, -wy),
        ca.horzcat(wy, -wz,   0,  wx),
        ca.horzcat(wz,  wy, -wx,   0)
    )

# Euler angle rates (approximate method)
# Mapping from body rates to Euler angle derivatives
def angularVel_to_EulerRates(roll,pitch,yaw,w):
    T = ca.SX.zeros(3,3)
    T[0,0] = 1
    T[1,1] = ca.cos(roll)
    T[1,2] = ca.sin(roll)*ca.tan(pitch)
    T[2,1] = -ca.sin(roll)
    T[2,2] = ca.cos(roll)*ca.tan(pitch)

    rpy_dot = ca.mtimes(T, w)
    return rpy_dot


#def getTrack():
#    track_file = os.path.join(str(Path(__file__).parent), "tracks/", track)
#    array=np.loadtxt(track_file, skiprows=1)
#    sref = array[1:,0]
#    xref = array[1:,1]
#    yref = array[1:,2]
#    zref = array[1:,3]
#
#    return sref, xref, yref, zref
#
#[s_ref, x_ref, y_ref, z_ref] = getTrack()
#
#length = len(s_ref)
#pathlength = s_ref[-1]


def check_stabilizzability(A, B):
    # Converti da CasADi a NumPy se necessario
    check=False
    if hasattr(A, 'full'):
        A = A.full()
    if hasattr(B, 'full'):
        B = B.full()

    n = A.shape[0]
    #orientamento ha 3 gdl ma quaternione a 4 componenti
    if A.shape[0]==13 :
        n=n-1

    # Matrice di controllabilità
    C = B
    for i in range(1, n):
        C = np.hstack((C, np.linalg.matrix_power(A, i) @ B))

    rank_C = matrix_rank(C)
    print(f"🔎 Rank controllabilità: {rank_C}/{n}")
    if rank_C == n:
        print("✅ Sistema CONTROLLABILE.")
        check=True
    else:
        print("⚠️  Sistema NON completamente controllabile.")

    # Autovalori di A
    eigs = eigvals(A)
    unstable_eigs = [eig for eig in eigs if np.real(eig) > 0]

    if not unstable_eigs:
        print("✅ Nessun autovalore instabile: sistema STABILIZZABILE.")
    else:
        print(f"Autovalori instabili: {unstable_eigs}")
        stabilizable = True
        for lam in unstable_eigs:
            test_mat = np.hstack([lam * np.eye(n) - A, B])
            rank = matrix_rank(test_mat)
            if rank < n:
                print(f"❌ Autovalore {lam:.3f} NON stabilizzabile.")
                stabilizable = False
            else:
                print(f"✅ Autovalore {lam:.3f} stabilizzabile.")

        if stabilizable:
            print("✅ Sistema STABILIZZABILE.")
            check=True
        else:
            print("❌ Sistema NON stabilizzabile.")
    
    # Plot autovalori
    plt.figure(figsize=(6,6))
    plt.axhline(0, color='black', lw=0.8)
    plt.axvline(0, color='black', lw=0.8)

    eigs = np.array(eigs)
    stable = eigs[np.real(eigs) <= 0]
    unstable = eigs[np.real(eigs) > 0]

    plt.scatter(np.real(stable), np.imag(stable), color='blue', label='Stabili (Re ≤ 0)')
    plt.scatter(np.real(unstable), np.imag(unstable), color='red', label='Instabili (Re > 0)')
    plt.xlabel('Parte Reale')
    plt.ylabel('Parte Immaginaria')
    plt.title('Autovalori di A')
    plt.legend()
    plt.grid(True)
    plt.axis('auto')
    plt.show()

    return check

def compute_terminal_cost_P(model, x_eq, u_eq, Q, R):
    """
    Calcola la matrice P del costo terminale risolvendo l'ARE continua linearizzando il modello ACADOS.

    Parameters:
    - model: modello ACADOS con attributi x, u, f_expl_expr (CasADi SX)
    - x_eq: punto di equilibrio stato (numpy array)
    - u_eq: punto di equilibrio input (numpy array)
    - Q, R: matrici di costo quadratiche (numpy array)

    Returns:
    - P: matrice del costo terminale (numpy array)
    """

    x = model.x
    u = model.u
    f = model.f_expl_expr

    # Jacobiane
    A_sym = ca.jacobian(f, x)
    B_sym = ca.jacobian(f, u)

    A_fun = ca.Function('A_fun', [x, u], [A_sym])
    B_fun = ca.Function('B_fun', [x, u], [B_sym])

    A = A_fun(x_eq, u_eq).full()
    B = B_fun(x_eq, u_eq).full()

    # Risolvo l'ARE continua
    P = solve_continuous_are(A, B, Q, R)

    return P


#def plot_drone(time, X, U, latexify=False, plt_show=True, time_label='$t$', x_labels=None, u_labels=None):
#    """
#    Params:
#        t: time values of the discretization
#        u_max: maximum absolute value(s) of u (scalar or list)
#        U: array with shape (N_sim-1, nu) or (N_sim, nu)
#        X_true: array with shape (N_sim, nx)
#        latexify: latex style plots
#    """
#    if latexify:
#        latexify_plot()
#
#    nx = X.shape[1]
#    nu = U.shape[1]
#    fig, axes = plt.subplots(nx + 1, 1, sharex=True, figsize=(8, 2 * (nx + 1)))
#
#
#    for i in range(nx):
#        axes[i].plot(t, X[:, i])
#        axes[i].grid()
#        axes[i].set_ylabel(x_labels[i] if x_labels else f'$x_{i}$')
#
#    # Ingressi
#    for i in range(nu):
#        axes[-1].step(t[:U.shape[0]], U[:, i], label=u_labels[i] if u_labels else f'$u_{i}$')
#    axes[-1].legend()
#
#    # limiti
#    if np.isscalar(u_max):
#        u_max_arr = [u_max] * nu
#    else:
#        u_max_arr = u_max
#
#    for i, umax in enumerate(u_max_arr):
#        axes[-1].hlines(umax, t[0], t[-1], linestyles='dashed', alpha=0.7)
#        axes[-1].hlines(-umax, t[0], t[-1], linestyles='dashed', alpha=0.7)
#
#    axes[-1].set_ylim([-1.2 * max(u_max_arr), 1.2 * max(u_max_arr)])
#    axes[-1].set_xlim(t[0], t[-1])
#    axes[-1].set_ylabel('Inputs')
#    axes[-1].set_xlabel(time_label)
#    axes[-1].grid()
#
#    plt.subplots_adjust(hspace=0.4)
#    fig.align_ylabels()
#
#    if plt_show:
#        plt.show()

def myPlotWithReference(time, ref, sim, labels, title, ncols=2):
    """
    Plotta confronto tra traiettorie di riferimento e simulate.
    """
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif"
    })

    if (np.ndim(ref) > 1) :
        n = ref.shape[1]
    else :
        n=1
    
    if n == 1:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(time, ref[:, 0], 'r--', label='Reference')
        ax.plot(time, sim[:, 0], 'b-', label='Simulation')
        ax.set_title(rf"${labels[0]}$", fontsize=12)
        ax.set_xlabel(r"Time [s]")
        ax.set_ylabel(rf"${labels[0]}$")
        ax.grid(True)
        ax.legend()
    else:
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4 * nrows))
        axes = axes.flatten()
        for i in range(n):
            axes[i].plot(time, ref[:, i], 'r--', label='Reference')
            axes[i].plot(time, sim[:, i], 'b-', label='Simulation')
            axes[i].set_title(rf"${labels[i]}$", fontsize=12)
            axes[i].set_xlabel(r"Time [s]")
            axes[i].set_ylabel(rf"${labels[i]}$")
            axes[i].grid(True)
            axes[i].legend()
        for j in range(n, nrows * ncols):
            fig.delaxes(axes[j])

    fig.suptitle(rf"\textbf{{{title}}}", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def myPlot(time, sim, labels, title, ncols=2):
    """
    Plotta traiettorie simulate.
    """
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif"
    })

    n = sim.shape[1]

    if n == 1:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(time, sim[:, 0], 'b-', label='Simulation')
        ax.set_title(rf"${labels[0]}$", fontsize=12)
        ax.set_xlabel(r"Time [s]")
        ax.set_ylabel(rf"${labels[0]}$")
        ax.grid(True)
        ax.legend()
    else:
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4 * nrows))
        axes = axes.flatten()
        for i in range(n):
            axes[i].plot(time, sim[:, i], 'b-', label='Simulation')
            axes[i].set_title(rf"${labels[i]}$", fontsize=12)
            axes[i].set_xlabel(r"Time [s]")
            axes[i].set_ylabel(rf"${labels[i]}$")
            axes[i].grid(True)
            axes[i].legend()
        for j in range(n, nrows * ncols):
            fig.delaxes(axes[j])

    fig.suptitle(rf"\textbf{{{title}}}", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


#def traj_plot3D_animated(t, *trajs, labels=None, colors=None, interval=30, step=2):
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
#
#    num_trajs = len(trajs)
#    if labels is None:
#        labels = [f'Trajectory {i+1}' for i in range(num_trajs)]
#    if colors is None:
#        colors = ['C'+str(i) for i in range(num_trajs)]
#
#    # Inizializza linee vuote
#    lines = []
#    for label, color in zip(labels, colors):
#        line, = ax.plot([], [], [], color=color, label=label, linewidth=2)
#        lines.append(line)
#
#    # Calcola limiti globali
#    all_xyz = np.concatenate(trajs, axis=0)
#    ax.set_xlim(np.min(all_xyz[:, 0]), np.max(all_xyz[:, 0]))
#    ax.set_ylim(np.min(all_xyz[:, 1]), np.max(all_xyz[:, 1]))
#    ax.set_zlim(np.min(all_xyz[:, 2]), np.max(all_xyz[:, 2]))
#
#    ax.set_xlabel('X')
#    ax.set_ylabel('Y')
#    ax.set_zlabel('Z')
#    ax.set_title('Animazione traiettorie 3D')
#    ax.legend()
#
#    def update(frame):
#        i = frame * step
#        i = min(i, len(t) - 1)
#        for line, traj in zip(lines, trajs):
#            line.set_data(traj[:i+1, 0], traj[:i+1, 1])
#            line.set_3d_properties(traj[:i+1, 2])
#        return lines
#
#    n_frames = len(t) // step + 1
#    ani = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=False)
#
#    plt.tight_layout()
#    plt.show()

def traj_plot3D_animated_with_orientation(t, drone_pos, drone_rot, obj_pos, obj_rot, interval=30, step=2):
    """
    Animazione 3D delle traiettorie di drone e oggetto, con assi di orientamento.

    Parametri:
    - t: array tempi (N,)
    - drone_pos: (N,3) posizioni drone
    - drone_rot: (N,3,3) matrici rotazione drone (ogni R[i] colonna = asse X,Y,Z)
    - obj_pos: (N,3) posizioni oggetto
    - obj_rot: (N,3,3) matrici rotazione oggetto
    - interval: intervallo animazione [ms]
    - step: passo frame
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Limiti globali per tutti i dati
    all_pos = np.vstack([drone_pos, obj_pos])
    ax.set_xlim(np.min(all_pos[:, 0]), np.max(all_pos[:, 0]))
    ax.set_ylim(np.min(all_pos[:, 1]), np.max(all_pos[:, 1]))
    ax.set_zlim(np.min(all_pos[:, 2]), np.max(all_pos[:, 2]))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Animazione con orientamento')
    ax.legend()

    # Linee traiettorie
    drone_line, = ax.plot([], [], [], 'r-', label='Drone Trajectory', linewidth=2)
    obj_line, = ax.plot([], [], [], 'b-', label='Object Trajectory', linewidth=2)

    # Linee per assi (3 per drone + 3 per oggetto)
    drone_axes_lines = [ax.plot([], [], [], color=c)[0] for c in ['r', 'g', 'b']]
    obj_axes_lines = [ax.plot([], [], [], color=c)[0] for c in ['r', 'g', 'b']]

    ax.legend()

    def plot_axes(origin, R, length=0.5):
        """
        Restituisce liste di punti per ogni asse da disegnare.
        """
        ends = origin[:,None] + R * length  # broadcasting: (3,3) * scalar
        # ends shape (3,3) = 3 vettori asse, colonne: assi X,Y,Z
        return [(origin, ends[:, i]) for i in range(3)]

    def update(frame):
        i = min(frame * step, len(t) - 1)

        # Aggiorna traiettorie
        drone_line.set_data(drone_pos[:i + 1, 0], drone_pos[:i + 1, 1])
        drone_line.set_3d_properties(drone_pos[:i + 1, 2])

        obj_line.set_data(obj_pos[:i + 1, 0], obj_pos[:i + 1, 1])
        obj_line.set_3d_properties(obj_pos[:i + 1, 2])

        # Aggiorna assi drone
        origin = drone_pos[i]
        R = RPY_to_R(drone_rot[i,0],drone_rot[i,1],drone_rot[i,2]).full()
        for idx, line in enumerate(drone_axes_lines):
            start, end = plot_axes(origin, R, length=1)[idx]
            line.set_data([start[0], end[0]], [start[1], end[1]])
            line.set_3d_properties([start[2], end[2]])

        # Aggiorna assi oggetto
        origin = obj_pos[i]
        R = RPY_to_R(obj_rot[i,0],obj_rot[i,1],obj_rot[i,1]).full()
        for idx, line in enumerate(obj_axes_lines):
            start, end = plot_axes(origin, R, length=1)[idx]
            line.set_data([start[0], end[0]], [start[1], end[1]])
            line.set_3d_properties([start[2], end[2]])

        # Ritorna linee per aggiornamento
        return [drone_line, obj_line] + drone_axes_lines + obj_axes_lines

    n_frames = len(t) // step + 1
    ani = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=False)

    plt.tight_layout()
    plt.show()
