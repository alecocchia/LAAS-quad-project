#!/usr/bin/env python3
import argparse, numpy as np
import matplotlib.pyplot as plt
import shutil, matplotlib as mpl

# =======================
#  Funzioni di plot
#  (aggiunte 2-3 piccole robustifiche)
# =======================
def myPlotWithReference(time, refs, sim, labels, title, ncols=2, use_tex=False):
    """
    Plotta confronto tra traiettorie di riferimento e simulate.

    - refs: lista di array, ciascuno shape (N, n) o (N,) --> riferimento
    - sim: array di shape (N, n) o (N,) --> simulazione
    - labels: lista di etichette per le variabili simulate (n)
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
                # Se il ref ha meno colonne, usa la prima
                col = ref[:, i] if ref.shape[1] > i else ref[:, 0]
                axes[i].plot(time, col, '--', color=ref_colors[j % len(ref_colors)], label=f"Ref {j+1}")
            axes[i].plot(time, sim[:, i], 'b-', label='Simulation')
            axes[i].set_title(labels[i], fontsize=12)
            axes[i].set_xlabel("Time [s]")
            axes[i].set_ylabel(labels[i])
            axes[i].grid(True)
            axes[i].legend()
        # rimuovi subplot extra
        for j in range(n, nrows * ncols):
            fig.delaxes(axes[j])

    fig.suptitle(title, fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def myPlot(time, sim, labels, title, ncols=2, use_tex=False):
    """
    Plotta traiettorie simulate.
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
    """
    Estrae da un file .npz dell'OCP le reference di posizione (x,y,z) e yaw.
    Assunzione: x_opt colonne [0:3] posizione, [6:10] quaternione (w,x,y,z).
    Ritorna (t, pos(N+1,3), yaw(N+1,))
    """
    t = ocp_npz['t'] if 't' in ocp_npz.files else np.arange(ocp_npz['x_opt'].shape[0])
    X = ocp_npz['x_opt']  # (N+1, nx)
    pos = X[:, 0:3]
    q = X[:, 6:10]        # (w,x,y,z)
    yaw = quat_to_yaw(q[:,1], q[:,2], q[:,3], q[:,0])
    return np.asarray(t), np.asarray(pos), np.asarray(yaw)

def main():
    ap = argparse.ArgumentParser(description="Plot dei log PID/refs da .npz")
    ap.add_argument("--log", type=str, default="/tmp/pid_run.npz",
                    help="File .npz prodotto dal RefSimLogger")
    ap.add_argument("--ocp", type=str, default=None,
                    help="(Opz.) File .npz con la soluzione OCP per sovrapporre la traiettoria ideale")
    ap.add_argument("--tex", action="store_true", help="Usa LaTeX nei plot")
    args = ap.parse_args()

    data = np.load(args.log)
    t = data['t'] if 't' in data.files else np.array([])
    # porta il tempo a partire da zero (per leggibilità)
    if t.size > 0:
        t = t - t[0]

    pref = data['pref']   if 'pref'   in data.files else np.empty((0,4))  # (N,4) [x,y,z,yaw]
    vref = data['vref']   if 'vref'   in data.files else np.empty((0,3))  # (N,3)
    p    = data['p']      if 'p'      in data.files else np.empty((0,4))  # (N,4) [x,y,z,yaw]
    v    = data['v']      if 'v'      in data.files else np.empty((0,3))  # (N,3)

    # (NUOVO) RPY e velocità angolari + riferimenti
    rpy      = data['rpy']      if 'rpy'      in data.files else np.empty((0,3))  # (N,3) [roll,pitch,yaw]
    omega    = data['omega']    if 'omega'    in data.files else np.empty((0,3))  # (N,3) [wx,wy,wz]
    pref_rpy = data['pref_rpy'] if 'pref_rpy' in data.files else np.empty((0,3))
    omegaref = data['omegaref'] if 'omegaref' in data.files else np.empty((0,3))

    wrench_cmd = data['wrench_cmd'] if 'wrench_cmd' in data.files else np.empty((0,4))
    wrench_ref = data['wrench_ref'] if 'wrench_ref' in data.files else np.empty((0,4))

    # Tronca tutto alla stessa lunghezza minima utile
    N = min([arr.shape[0] for arr in [t, pref, p, v, rpy, omega] if arr.size > 0] + [t.shape[0]])
    t = t[:N]
    if pref.size:      pref      = pref[:N]
    if p.size:         p         = p[:N]
    if v.size:         v         = v[:N]
    if vref.size:      vref      = vref[:N]
    if rpy.size:       rpy       = rpy[:N]
    if pref_rpy.size:  pref_rpy  = pref_rpy[:N]
    if omega.size:     omega     = omega[:N]
    if omegaref.size:  omegaref  = omegaref[:N]
    if wrench_cmd.size: wrench_cmd = wrench_cmd[:N]
    if wrench_ref.size: wrench_ref = wrench_ref[:N]

    print("t:", t.shape, "pref:", pref.shape, "p:", p.shape, "v:", v.shape, "wrench:", wrench_cmd.shape, "wrench_ref:", wrench_ref.shape)
    if p.size:
        print("p stats:", np.nanmin(p, axis=0), np.nanmax(p, axis=0))
    if v.size:
        print("v stats:", np.nanmin(v, axis=0), np.nanmax(v, axis=0))
    if wrench_cmd.size:
        print("wrench stats:", np.nanmin(wrench_cmd, axis=0), np.nanmax(wrench_cmd, axis=0))

    # --- POSIZIONE ---
    if p.size and pref.size:
        myPlotWithReference(t, [pref[:, :3]], p[:, :3],
                            labels=[r"$x$", r"$y$", r"$z$"],
                            title="Position: reference vs simulation",
                            ncols=3, use_tex=args.tex)
    elif p.size:
        myPlot(t, p[:, :3], labels=[r"$x$", r"$y$", r"$z$"],
               title="Position: simulation", ncols=3, use_tex=args.tex)

    # --- VELOCITÀ ---
    # Se vref contiene NaN (logger quando non c'è twist), plottiamo solo v.
    if v.size:
        if vref.size and not np.isnan(vref).all():
            myPlotWithReference(t, [vref], v,
                                labels=[r"$v_x$", r"$v_y$", r"$v_z$"],
                                title="Velocity: reference vs simulation",
                                ncols=3, use_tex=args.tex)
        else:
            myPlot(t, v, labels=[r"$v_x$", r"$v_y$", r"$v_z$"],
                   title="Velocity: simulation", ncols=3, use_tex=args.tex)

    # --- (NUOVO) ASSETTO RPY ---
    if rpy.size:
        if pref_rpy.size and not np.isnan(pref_rpy).all():
            myPlotWithReference(t, [pref_rpy], rpy,
                                labels=[r"$\phi$ [rad]", r"$\theta$ [rad]", r"$\psi$ [rad]"],
                                title="Attitude (RPY): reference vs simulation",
                                ncols=3, use_tex=args.tex)
        else:
            myPlot(t, rpy,
                   labels=[r"$\phi$ [rad]", r"$\theta$ [rad]", r"$\psi$ [rad]"],
                   title="Attitude (RPY): simulation", ncols=3, use_tex=args.tex)

    # --- (NUOVO) VELOCITÀ ANGOLARI ---
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

    # --- YAW ---
    if p.size and pref.size:
        myPlotWithReference(t, [pref[:, 3]], p[:, 3],
                            labels=[r"$\psi$ [rad]"],
                            title="Yaw: reference vs simulation",
                            ncols=1, use_tex=args.tex)
    elif p.size:
        myPlot(t, p[:, 3:4], labels=[r"$\psi$ [rad]"],
               title="Yaw: simulation", ncols=1, use_tex=args.tex)

    # --- WRENCH ---
    if wrench_cmd.size and wrench_ref.size and wrench_cmd.shape[0] == wrench_ref.shape[0]:
        myPlotWithReference(
            np.linspace(0, t[-1] if t.size else wrench_cmd.shape[0], wrench_cmd.shape[0]),
            [wrench_ref],
            wrench_cmd,
            labels=[r"$F_z$", r"$\tau_x$", r"$\tau_y$", r"$\tau_z$"],
            title="Wrench: reference vs commanded",
            ncols=2, use_tex=args.tex
        )
    elif wrench_cmd.size:
        myPlot(
            np.linspace(0, t[-1] if t.size else wrench_cmd.shape[0], wrench_cmd.shape[0]),
            wrench_cmd,
            labels=[r"$F_z$", r"$\tau_x$", r"$\tau_y$", r"$\tau_z$"],
            title="Wrench (commanded)",
            ncols=2, use_tex=args.tex
        )

    # --- (Opzionale) OCP IDEALE SOVRAPPOSTO ---
    if args.ocp:
        ocp = np.load(args.ocp)
        tocp, pos_ocp, yaw_ocp = extract_ocp_refs(ocp)
        # riallinea a lunghezza comune per plot rapido
        tocp = tocp - tocp[0]
        K = min(len(t), len(tocp), len(pos_ocp))
        if K > 10:  # solo se abbiamo abbastanza punti
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