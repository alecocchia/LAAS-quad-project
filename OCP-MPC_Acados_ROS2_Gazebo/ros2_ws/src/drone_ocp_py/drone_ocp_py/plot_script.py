#!/usr/bin/env python3
import argparse, numpy as np
import matplotlib.pyplot as plt
import shutil, matplotlib as mpl
import os
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation

# =======================
#  Funzioni di plot
# =======================
def myPlotWithReference(time, refs, sim, labels, title, ncols=2, use_tex=True):
    """
    Plotta confronto tra traiettorie di riferimento e simulate.
    """
    import numpy as np
    plt.rcParams.update({"text.usetex": use_tex, "font.family": "serif"})

    time = np.asarray(time).reshape(-1)
    sim = np.asarray(sim)
    if sim.ndim == 1:
        sim = sim[:, np.newaxis]
    N, n = sim.shape
    assert len(labels) >= n, "labels deve avere almeno n voci."

    refs = [np.asarray(r) for r in refs]
    # forza refs a 2D e tronca a N
    refs = [r[:, None] if r.ndim == 1 else r for r in refs]
    refs = [r[:N] for r in refs]
    time = time[:N]

    ref_colors = ['r', 'g', 'm', 'c', 'y', 'k']

    if n == 1:
        fig, ax = plt.subplots(figsize=(10, 4))
        for j, ref in enumerate(refs):
            ax.plot(time, ref[:, 0], '--', color=ref_colors[j % len(ref_colors)], label=f"Ref {j+1}")
        ax.plot(time, sim[:, 0], 'b-', label='Simulation')
        ax.set_title(labels[0], fontsize=12)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(labels[0])
        ax.grid(True)
        ax.legend()
    else:
        ncols = max(1, ncols)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4 * nrows))
        axes = axes.flatten()
        for i in range(n):
            for j, ref in enumerate(refs):
                col = ref[:, i] if ref.shape[1] > i else ref[:, 0]
                axes[i].plot(time, col, '--', color=ref_colors[j % len(ref_colors)], label=f"Ref {j+1}")
            axes[i].plot(time, sim[:, i], 'b-', label='Simulation')
            axes[i].set_title(labels[i], fontsize=12)
            axes[i].set_xlabel("Time [s]")
            axes[i].set_ylabel(labels[i])
            axes[i].grid(True)
            axes[i].legend()
        for j in range(n, nrows * ncols):
            fig.delaxes(axes[j])

    fig.suptitle(title, fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def myPlot(time, sim, labels, title, ncols=2, use_tex=True):
    """
    Plotta traiettorie simulate o array singoli.
    """
    import numpy as np
    plt.rcParams.update({"text.usetex": use_tex, "font.family": "serif"})

    time = np.asarray(time).reshape(-1)
    sim = np.asarray(sim)
    if sim.ndim == 1:
        sim = sim[:, None]
    N, n = sim.shape
    assert len(labels) >= n, "labels deve avere almeno n voci."

    time = time[:N]

    if n == 1:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(time, sim[:, 0], 'b-', label='Simulation')
        ax.set_title(labels[0], fontsize=12)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(labels[0])
        ax.grid(True)
        ax.legend()
    else:
        ncols = max(1, ncols)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4 * nrows))
        axes = axes.flatten()
        for i in range(n):
            axes[i].plot(time, sim[:, i], 'b-', label='Simulation')
            axes[i].set_title(labels[i], fontsize=12)
            axes[i].set_xlabel("Time [s]")
            axes[i].set_ylabel(labels[i])
            axes[i].grid(True)
            axes[i].legend()
        for j in range(n, nrows * ncols):
            fig.delaxes(axes[j])

    fig.suptitle(title, fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# =======================
#  Utility
# =======================
def quat_to_yaw(x, y, z, w):
    siny_cosp = 2.0*(w*z + x*y)
    cosy_cosp = 1.0 - 2.0*(y*y + z*z)
    return np.arctan2(siny_cosp, cosy_cosp)

def extract_ocp_refs(ocp_npz):
    t = ocp_npz['t'] if 't' in ocp_npz.files else np.arange(ocp_npz['x_opt'].shape[0])
    X = ocp_npz['x_opt']
    pos = X[:, 0:3]
    q = X[:, 6:10]
    yaw = quat_to_yaw(q[:,1], q[:,2], q[:,3], q[:,0])
    return np.asarray(t), np.asarray(pos), np.asarray(yaw)

def get_camera_offset_from_sdf(sdf_path=None):
    """
    Legge dinamicamente l'offset della camera dal file model.sdf.
    Ritorna un array numpy [x, y, z].
    """
    cam_offset = np.array([0.0, 0.0, 0.0]) # Default
    
    # 1. Trova l'SDF tramite ament_index_python se non passato
    if sdf_path is None:
        try:
            from ament_index_python.packages import get_package_share_directory
            pkg_share_dir = get_package_share_directory('mrsim_gazebo_sim')
            sdf_path = os.path.join(pkg_share_dir, 'models', 'mrsim-quad-unico', 'model.sdf')
        except ImportError:
            print("[WARNING] ROS 2 workspace not sourced. Provide --sdf argument. Using default [0,0,0].")
            return cam_offset

    # 2. Parsing del file XML
    if os.path.exists(sdf_path):
        try:
            tree = ET.parse(sdf_path)
            root = tree.getroot()
            # Cerca il tag sensor con name="camera"
            camera_sensor = root.find('.//sensor[@name="camera"]')
            if camera_sensor is not None:
                pose_tag = camera_sensor.find('pose')
                if pose_tag is not None and pose_tag.text:
                    coords = pose_tag.text.strip().split()
                    if len(coords) >= 3:
                        cam_offset = np.array([float(coords[0]), float(coords[1]), float(coords[2])])
                        print(f"[INFO] Camera offset letto dinamicamente dall'SDF: {cam_offset}")
                        return cam_offset
        except Exception as e:
            print(f"[WARNING] Errore nel parsing dell'SDF per la camera: {e}")
            
    print("[INFO] Uso offset della camera di default: [0, 0, 0]")
    return cam_offset


def main():
    ap = argparse.ArgumentParser(description="Plot dei log PID/refs da .npz")
    ap.add_argument("--log", type=str, default="/tmp/pid_run.npz",
                    help="File .npz prodotto dal RefSimLogger")
    ap.add_argument("--ocp", type=str, default=None,
                    help="(Opz.) File .npz con la soluzione OCP per sovrapporre la traiettoria ideale")
    ap.add_argument("--sdf", type=str, default=None,
                    help="(Opz.) Percorso manuale al file model.sdf per l'offset della camera")
    ap.add_argument("--tex", action="store_true", help="Usa LaTeX nei plot")
    args = ap.parse_args()

    data = np.load(args.log)
    t = data['t'] if 't' in data.files else np.array([])
    if t.size > 0:
        t = t - t[0]

    pref = data['pref']   if 'pref'   in data.files else np.empty((0,4))
    vref = data['vref']   if 'vref'   in data.files else np.empty((0,3))
    p    = data['p']      if 'p'      in data.files else np.empty((0,4))
    v    = data['v']      if 'v'      in data.files else np.empty((0,3))
    q    = data['q']      if 'q'      in data.files else np.empty((0,4)) 

    rpy      = data['rpy']      if 'rpy'      in data.files else np.empty((0,3))
    omega    = data['omega']    if 'omega'    in data.files else np.empty((0,3))
    pref_rpy = data['pref_rpy'] if 'pref_rpy' in data.files else np.empty((0,3))
    omegaref = data['omegaref'] if 'omegaref' in data.files else np.empty((0,3))

    wrench_cmd = data['wrench_cmd'] if 'wrench_cmd' in data.files else np.empty((0,4))
    wrench_ref = data['wrench_ref'] if 'wrench_ref' in data.files else np.empty((0,4))

    peg_pos = data['peg_pos'] if 'peg_pos' in data.files else np.empty((0,3))
    online_ref = data['online_ref'] if 'online_ref' in data.files else np.empty((0,6))
    online_visual_ref = data['online_visual_ref'] if 'online_visual_ref' in data.files else np.empty((0,2))

    N = min([arr.shape[0] for arr in [t, pref, p, v, q, rpy, omega, peg_pos, online_ref] if arr.size > 0] + [t.shape[0]])
    t = t[:N]
    if pref.size:      pref      = pref[:N]
    if p.size:         p         = p[:N]
    if v.size:         v         = v[:N]
    if q.size:         q         = q[:N]
    if vref.size:      vref      = vref[:N]
    if rpy.size:       rpy       = rpy[:N]
    if pref_rpy.size:  pref_rpy  = pref_rpy[:N]
    if omega.size:     omega     = omega[:N]
    if omegaref.size:  omegaref  = omegaref[:N]
    if wrench_cmd.size: wrench_cmd = wrench_cmd[:N]
    if wrench_ref.size: wrench_ref = wrench_ref[:N]
    if peg_pos.size: peg_pos = peg_pos[:N]
    if online_ref.size: online_ref = online_ref[:N]
    if online_visual_ref.size: online_visual_ref = online_visual_ref[:N]

    # === LETTURA DINAMICA OFFSET DALL'SDF ===
    cam_offset = get_camera_offset_from_sdf(args.sdf)

    # === CALCOLI GEOMETRICI & ERROR NORMS ===
    actual_dist = np.empty((0,))
    desired_pos = np.empty((0,3))
    Y_c = np.empty((0,))
    Z_c = np.empty((0,))

    err_pos_xy = np.empty((0,))
    err_visual = np.empty((0,))
    err_vel = np.empty((0,))
    err_rp = np.empty((0,))
    err_omega = np.empty((0,))
    err_u = np.empty((0,))

    if p.size and peg_pos.size and online_ref.size and q.size:
        # Posa Cartesiana Desiderata
        r = online_ref[:, 0]
        pan = online_ref[:, 1]
        tilt = online_ref[:, 2]

        des_x = peg_pos[:, 0] + r * np.cos(tilt) * np.cos(pan)
        des_y = peg_pos[:, 1] + r * np.cos(tilt) * np.sin(pan)
        des_z = peg_pos[:, 2] + r * np.sin(tilt)
        desired_pos = np.column_stack((des_x, des_y, des_z))

        # Calcolo Posa della Camera applicando l'offset dinamico
        q_scipy = np.column_stack((q[:, 1], q[:, 2], q[:, 3], q[:, 0]))
        rots = Rotation.from_quat(q_scipy)

        p_cam = p[:, :3] + rots.apply(cam_offset)
        
        # Errore Visual Servoing (Y_c, Z_c)
        p_rel_world = peg_pos - p_cam
        P_c = rots.inv().apply(p_rel_world)
        Y_c = P_c[:, 1]
        Z_c = P_c[:, 2]

        # Raggio effettivo dalla TELECAMERA al bersaglio
        actual_dist = np.linalg.norm(p_cam[:, :3] - peg_pos, axis=1)

        # =======================================================
        # CALCOLO NORME ERRORI (yref - y_expr)
        # =======================================================
        if wrench_cmd.size and v.size and omega.size:
            err_pos_xy = np.linalg.norm(p_cam[:, :2] - desired_pos[:, :2], axis=1)
            err_visual = np.linalg.norm(np.column_stack((Y_c, Z_c)), axis=1)
            err_vel = np.linalg.norm(v, axis=1)
            err_rp = np.linalg.norm(q[:, 1:3], axis=1)
            err_omega = np.linalg.norm(omega, axis=1)
            
            m = 1.28
            g = 9.81
            delta_u = wrench_cmd.copy()
            delta_u[:, 0] -= (m * g)
            err_u = np.linalg.norm(delta_u, axis=1)


    # --- PLOT NORME ERRORI (yref - y_expr) ---
    if err_u.size > 0:
        error_matrix = np.column_stack((err_pos_xy, err_visual, err_vel, err_rp, err_omega, err_u))
        error_labels = [
            r"$||e_{pos_{XY}}|| \ [m]$", 
            r"$||e_{visual}|| \ [m]$", 
            r"$||e_{vel}|| \ [m/s]$", 
            r"$||e_{rp}||$", 
            r"$||e_{\omega}|| \ [rad/s]$", 
            r"$||e_u||$"
        ]
        myPlot(
            t,
            error_matrix,
            labels=error_labels,
            title="Norme degli Errori dell'MPC (yref - y_expr)",
            ncols=3,
            use_tex=args.tex
        )

    # --- POSIZIONE ---
    if p.size and pref.size:
        myPlotWithReference(t, [pref[:, :3]], p[:, :3],
                            labels=[r"$x$", r"$y$", r"$z$"],
                            title="Position: reference vs simulation",
                            ncols=3, use_tex=args.tex)
    elif p.size:
        myPlot(t, p[:, :3], labels=[r"$x$", r"$y$", r"$z$"],
               title="Position: simulation", ncols=3, use_tex=args.tex)

    # --- 1. PLOT DELLA DISTANZA RADIALE DALLA CAMERA ---
    if actual_dist.size and not np.isnan(actual_dist).all():
        myPlotWithReference(
            t, 
            [online_ref[:, 0]],  
            actual_dist,         
            labels=[r"$Radius\ [m]$"],
            title="Tracking: Distanza ottica reale vs Distanza Desiderata",
            ncols=1, use_tex=args.tex
        )

    # --- 2. PLOT DELLA POSIZIONE CARTESIANA ASSOLUTA (Drone vs Target) ---
    if desired_pos.size and not np.isnan(desired_pos).all():
        myPlotWithReference(
            t, 
            [desired_pos],       
            p[:, :3],            
            labels=[r"$X\ [m]$", r"$Y\ [m]$", r"$Z\ [m]$"],
            title="Tracking: Posa Reale Drone vs Posa Target",
            ncols=3, use_tex=args.tex
        )
        
    print(online_visual_ref.shape)
    # --- 3. PLOT DEGLI ERRORI VISIVI (Y_c, Z_c) ---
    if Y_c.size and not np.isnan(Y_c).all():
        myPlotWithReference(
            t,
            [online_visual_ref],  
            np.column_stack((Y_c, Z_c)),                
            labels=[r"$Y_c\ [m]$", r"$Z_c\ [m]$"],
            title="Visual Servoing: Errore sul piano immagine (Y_c, Z_c)",
            ncols=2, use_tex=args.tex
        )

    # --- VELOCITÀ ---
    if v.size:
        if vref.size and not np.isnan(vref).all():
            myPlotWithReference(t, [vref], v,
                                labels=[r"$v_x$", r"$v_y$", r"$v_z$"],
                                title="Velocity: reference vs simulation",
                                ncols=3, use_tex=args.tex)
        else:
            myPlot(t, v, labels=[r"$v_x$", r"$v_y$", r"$v_z$"],
                   title="Velocity: simulation", ncols=3, use_tex=args.tex)

    # --- ASSETTO RPY ---
    if rpy.size:
        if pref_rpy.size and not np.isnan(pref_rpy).all():
            myPlotWithReference(t, [np.unwrap(pref_rpy,axis=0)], np.unwrap(rpy,axis=0),
                                labels=[r"$\phi$ [rad]", r"$\theta$ [rad]", r"$\psi$ [rad]"],
                                title="Attitude (RPY): reference vs simulation",
                                ncols=3, use_tex=args.tex)
        else:
            myPlot(t, np.unwrap(rpy),
                   labels=[r"$\phi$ [rad]", r"$\theta$ [rad]", r"$\psi$ [rad]"],
                   title="Attitude (RPY): simulation", ncols=3, use_tex=args.tex)

    # --- VELOCITÀ ANGOLARI ---
    if omega.size:
        if omegaref.size and not np.isnan(omegaref).all():
            myPlotWithReference(t, [omegaref], omega,
                                labels=[r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"],
                                title="Angular rates: reference vs simulation",
                                ncols=3, use_tex=args.tex)
        else:
            myPlot(t, omega,
                   labels=[r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"],
                   title="Angular rates: simulation", ncols=3, use_tex=args.tex)

    # --- YAW IDEALE TEORICO ---
    if p.size and peg_pos.size:
        yaw_ideal = np.arctan2(peg_pos[:, 1] - p[:, 1], peg_pos[:, 0] - p[:, 0])
        myPlotWithReference(t, [np.unwrap(yaw_ideal)], np.unwrap(p[:, 3]),
                            labels=[r"$\psi$ [rad]"],
                            title="Yaw: Ideal (pointing to peg) vs Simulation",
                            ncols=1, use_tex=args.tex)
    elif p.size:
        myPlot(t, p[:, 3:4], labels=[r"$\psi$ [rad]"],
               title="Yaw: simulation", ncols=1, use_tex=args.tex)

    # --- WRENCH IDEALE TEORICO (HOVERING) ---
    if wrench_cmd.size:
        m_drone = 1.28
        g_accel = 9.81
        wrench_ref_ideal = np.zeros_like(wrench_cmd)
        wrench_ref_ideal[:, 0] = m_drone * g_accel
        
        myPlotWithReference(
            np.linspace(0, t[-1] if t.size else wrench_cmd.shape[0], wrench_cmd.shape[0]),
            [wrench_ref_ideal],
            wrench_cmd,
            labels=[r"$F_z$", r"$\tau_x$", r"$\tau_y$", r"$\tau_z$"],
            title="Wrench: Reference (Hovering) vs Commanded",
            ncols=2, use_tex=args.tex
        )

    # --- OCP IDEALE SOVRAPPOSTO ---
    if args.ocp:
        ocp = np.load(args.ocp)
        tocp, pos_ocp, yaw_ocp = extract_ocp_refs(ocp)
        tocp = tocp - tocp[0]
        K = min(len(t), len(tocp), len(pos_ocp))
        if K > 10: 
            myPlotWithReference(t[:K], [pos_ocp[:K]], p[:K, :3] if p.size else pos_ocp[:K],
                                labels=[r"$x$", r"$y$", r"$z$"],
                                title="Position: OCP (ideal) vs received/simulated",
                                ncols=3, use_tex=args.tex)
            if p.size:
                myPlotWithReference(t[:K], [yaw_ocp[:K]], p[:K, 3],
                                    labels=[r"$\psi$ [rad]"],
                                    title="Yaw: OCP (ideal) vs simulation",
                                    ncols=1, use_tex=args.tex)

if __name__ == "__main__":
    main()