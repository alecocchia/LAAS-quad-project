#planner
import casadi as ca
import numpy as np


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

def generate_trapezoidal_trajectory(x0, x_ref, t0, tf, dt, v_max=1.0, a_max=1.0):
    """
    Genera una traiettoria p(t), rpy(t) con profilo trapezoidale lungo p_f - p_in e interpolazione lineare in rpy.
    
    Args:
        p_in: posizione iniziale (3,)
        p_f: posizione finale (3,)
        rpy_in: rotazione iniziale in RPY (3,)
        rpy_f: rotazione finale in RPY (3,)
        t0: tempo iniziale
        tf: tempo finale
        dt: passo di campionamento
        v_max: velocità massima normalizzata lungo la curvilinea s
        a_max: accelerazione massima normalizzata lungo s
        
    Returns:
        Trajectory: oggetto contenente t_vec, p_func(t), rpy_func(t)
    """

    p_in=x0[0:3]
    rpy_in=x0[3:6]
    p_f=x_ref[0:3]
    rpy_f=x_ref[3:6]

    dp = np.array(p_f) - np.array(p_in)
    L = np.linalg.norm(dp)
    if L == 0:
        raise ValueError("Punti iniziale e finale coincidenti.")

    # Tempo
    t_vec = np.arange(t0, tf + dt, dt)
    T = tf - t0
    t_sym = ca.SX.sym('t')

    # Trapezoidal profile s(t)
    t_ramp = v_max / a_max
    if T < 2 * t_ramp:
        t_ramp = T / 2
        v_max = a_max * t_ramp

    def s_trapezoid_expr(t):
        s1 = 0.5 * a_max * t_ramp**2
        t2 = T - t_ramp
        s2 = s1 + v_max * (t2 - t_ramp)

        s = ca.if_else(
            t < t_ramp,
            0.5 * a_max * t**2,
            ca.if_else(
                t < t2,
                s1 + v_max * (t - t_ramp),
                s2 + v_max * (t - t2) - 0.5 * a_max * (t - t2)**2
            )
        )
        s_total = s2 + v_max * t_ramp - 0.5 * a_max * t_ramp**2  # = L in teoria
        s_norm = ca.fmin(s / s_total, 1.0)  # Clamp a 1.0 per sicurezza
        return s_norm


    s_expr = s_trapezoid_expr(t_sym)
    s_func = ca.Function('s', [t_sym], [s_expr])

    # Posizione
    p_expr = ca.vertcat(*[p_in[i] + s_expr * (p_f[i] - p_in[i]) for i in range(3)])
    p_func = ca.Function('p_t', [t_sym], [p_expr])

    # Rotazione (RPY)
    rpy_expr = ca.vertcat(*[rpy_in[i] + s_expr * (rpy_f[i] - rpy_in[i]) for i in range(3)])
    rpy_func = ca.Function('rpy_t', [t_sym], [rpy_expr])

    p_vals = np.array([p_func(t).full().flatten() for t in t_vec])
    rpy_vals = np.array([rpy_func(t).full().flatten() for t in t_vec])
    return (t_vec, p_vals, rpy_vals)



def traj_plot3D_animated(t, *trajs, labels=None, colors=None, interval=30, step=2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    num_trajs = len(trajs)
    if labels is None:
        labels = [f'Trajectory {i+1}' for i in range(num_trajs)]
    if colors is None:
        colors = ['C'+str(i) for i in range(num_trajs)]

    # Inizializza linee vuote
    lines = []
    for label, color in zip(labels, colors):
        line, = ax.plot([], [], [], color=color, label=label, linewidth=2)
        lines.append(line)

    # Calcola limiti globali
    all_xyz = np.concatenate(trajs, axis=0)
    ax.set_xlim(np.min(all_xyz[:, 0]), np.max(all_xyz[:, 0]))
    ax.set_ylim(np.min(all_xyz[:, 1]), np.max(all_xyz[:, 1]))
    ax.set_zlim(np.min(all_xyz[:, 2]), np.max(all_xyz[:, 2]))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Animazione traiettorie 3D')
    ax.legend()

    def update(frame):
        i = frame * step
        i = min(i, len(t) - 1)
        for line, traj in zip(lines, trajs):
            line.set_data(traj[:i+1, 0], traj[:i+1, 1])
            line.set_3d_properties(traj[:i+1, 2])
        return lines

    n_frames = len(t) // step + 1
    ani = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=False)

    plt.tight_layout()
    plt.show()

