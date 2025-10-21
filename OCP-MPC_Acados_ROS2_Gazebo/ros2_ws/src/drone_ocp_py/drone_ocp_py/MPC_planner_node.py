#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPC_planner_node.py — PLANNER ONLY (gemello I/O di ocp_planner_node)
- Nessuna pubblicazione comandi controllo.
- peg_path_callback salva SOLO il path del peg (p_obj, rpy_obj).
- Configurazione MPC in configure_mpc(), chiamata allo start (/peg_pose).
- Risoluzione MPC in solve_MPC(xk), richiamata nel timer.
- human_goal aggiorna dinamicamente il riferimento mutual.

Dipendenze progetto: drone_MPC_settings.py, MPC_main.py, common.py
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from geometry_msgs.msg import PoseStamped, TwistStamped, TransformStamped, Wrench
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Bool, String

import numpy as np
import casadi as ca
from casadi import pi as pi

from drone_ocp_py.drone_MPC_settings import (
    setup_model, setup_initial_conditions, configure_mpc, set_initial_state, build_yref_online, build_yref_terminal
)
from drone_ocp_py.common import quat_to_RPY, RPY_to_quat, R_to_RPY, RPY_to_R

import tf2_ros


class MpcPlannerNode(Node):
    def __init__(self):
        super().__init__('mpc_planner_node')

        # === Modello e condizioni iniziali (coerenti con OCP) ===
        self.model, self.model_rpy = setup_model()
        self.x0, self.x0_rpy = setup_initial_conditions()

        # === Tempo/Orizzonte (coerenti con OCP) ===
        self.Tf = 20.0
        self.Tp = 2 # tempo di predizione (finestra MPC)
        self.ts = 0.02
        self.ts_peg = 0.005
        self.N_horiz = int(self.Tf / self.ts)

        self.t_prev = 0.0

        # === Stato MPC / loop ===
        self.mpc_ready = False         # solver configurato
        self.path_received = False     # ricevuto peg_path
        self.start_received = False    # ricevuto /peg_pose (trigger)
        self.k = 0
        self.mpc_path_published = False

        self.u_prev = None
        self.x_prev = None
        self.last_u0 = None

        # === Dati target da /peg_path ===
        self.p_obj = None
        self.rpy_obj = None

        # === Riferimenti mutual iniziali (come OCP) ===
        radius = 2.0
        mut_pos_ref = np.array([radius, 0.0, 0.0])   # [r, pan, tilt]
        mut_rot_ref = np.array([0.0, 0.0, pi/2])     # rpy
        mut_pos_final_ref = np.array([radius, 0.0, 0.0])
        mut_rot_final_ref = np.array([0.0, 0.0, pi])

        self.ref = np.concatenate([mut_pos_ref, mut_rot_ref])
        self.final_ref = np.concatenate([mut_pos_final_ref, mut_rot_final_ref])
        self.current_ref = self.ref.copy()  # aggiornabile via /human_goal

        # --- parametro: durata override umana (s) ---
        self.declare_parameter('human_hold_ref', 2.0)
        self.declare_parameter('control_flag',  1)  # 1 -> MPC controller on, 0 -> MPC controller off

        control_flag = self.get_parameter('control_flag').get_parameter_value().integer_value
        print(control_flag)
        wrench_topic_name = '/wrench_cmd' if control_flag == 1 else '/optimal_wrench'
        print("MPC pubblica su:", wrench_topic_name)


        # --- reference base (statico iniziale) e stato human override ---
        self.base_ref = self.ref.copy()      # riferimento base quando non c'è override
        self.hgoal_ref = None                # ultimo human goal convertito in [r,pan,tilt,roll,pitch,yaw]
        self.hgoal_until = None              # rclpy.time.Time di scadenza override

        # === Stato corrente (per dynamic MPC/TF/visual) ===
        self.current_position = np.zeros(3)
        self.current_rpy = np.zeros(3)
        self.current_vel = np.zeros(3)
        self.current_ang_vel = np.zeros(3)

        # === Publisher latched ===
        qos_latched = QoSProfile(
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            depth=1
        )
        self.ready_publisher  = self.create_publisher(Bool, '/drone_planner_ready',  qos_latched)
        self.optimal_path_pub = self.create_publisher(Path, '/optimal_drone_path', qos_latched)

        # === Subscriber latched del peg_path ===
        self.peg_path_subscription = self.create_subscription(
            Path, '/peg_path', self.peg_path_callback, qos_latched)

        # === Odometry (per TF/visual) ===
        self.odom_subscription = self.create_subscription(Odometry, '/odometry', self.odom_callback, 10)

        # === Pubblicazioni realtime (pose/twist predetti) ===
        self.single_pose_pub  = self.create_publisher(PoseStamped,  '/optimal_drone_pose',  1)
        self.single_twist_pub = self.create_publisher(TwistStamped, '/optimal_drone_twist', 1)
        self.single_wrench_pub = self.create_publisher(Wrench, wrench_topic_name, 1)
        self.tf_broadcaster   = tf2_ros.TransformBroadcaster(self)

        # === Trigger di start ( come nell’OCP) ===
        self.control_timer = None
        self.start_subscription = self.create_subscription(PoseStamped, '/peg_pose', self.start_callback, 10)

        # === Goal umano per ref dinamico ===
        self.human_goal_sub = self.create_subscription(PoseStamped, 'human_goal', self.human_goal_callback, 10)

        self.get_logger().info("MPC Planner Node (planner-only) avviato. In attesa di /peg_path e /peg_pose.")

    # ==================== Callbacks I/O ====================

    def peg_path_callback(self, msg: Path):
        """
        Ricezione path peg → salva p_obj, rpy_obj.
        """
        p_obj_list, rpy_obj_list = [], []
        count = 0
        times_ratio = max(1, int(round(self.ts / self.ts_peg)))

        for pose_stamped in msg.poses:
            if count % times_ratio == 0:
                p = pose_stamped.pose.position
                q = pose_stamped.pose.orientation
                rpy = quat_to_RPY([q.w, q.x, q.y, q.z])  # w,x,y,z
                p_obj_list.append([p.x, p.y, p.z])
                rpy_obj_list.append(np.squeeze(np.array(rpy)))
            count += 1

        self.p_obj = np.array(p_obj_list)
        self.rpy_obj = np.squeeze(np.array(rpy_obj_list))
        self.path_received = True
        self.get_logger().info(f"peg_path ricevuto. M={len(self.p_obj)} campioni.")

        # Configura MPC
        self.configure_mpc()
        self.mpc_ready = True
        # comunicazione segnale ready al peg_planner per inizio traiettoria
        self.ready_publisher.publish(Bool(data=True))
        self.get_logger().info("MPC planner ready")

        # (opzionale) pubblica subito un path latched “vuoto” (warm-start) se già configurato
        if self.mpc_ready and self.x_prev is not None:
            self.publish_predicted_path_from_buffers()

    def start_callback(self, _msg: PoseStamped):
        """
        Start del planner: se path ricevuto, configura MPC e avvia il timer.
        (Speculare a ocp_planner_node: lo start serve da trigger)
        """
        if self.start_received:
            return

        if not self.path_received:
            self.get_logger().warn("Start ricevuto ma peg_path non ancora disponibile. Attendo path...")
            return

        self.start_received = True

        # Avvia ciclo
        self.get_logger().info("Start effetuato. Avvio ciclo MPC (planner-only).")
        self.control_timer = self.create_timer(self.ts, self.control_step)
        # opzionale: rimozione del subscriber di start
        self.destroy_subscription(self.start_subscription)


    def odom_callback(self, msg: Odometry):
        """Aggiorna stato corrente per TF/visual."""
        self.current_position[:] = [msg.pose.pose.position.x,
                                    msg.pose.pose.position.y,
                                    msg.pose.pose.position.z]
        self.current_rpy[:] = np.array(
            quat_to_RPY([msg.pose.pose.orientation.w,
                         msg.pose.pose.orientation.x,
                         msg.pose.pose.orientation.y,
                         msg.pose.pose.orientation.z]).full()
        ).squeeze()

        self.R = np.array(RPY_to_R(self.current_rpy[0],self.current_rpy[1],self.current_rpy[2]).full())

        # Velocità dall'odometria:
        # NB: in molte configurazioni Gazebo dà twist in frame del child (body).
        # Siccome nel mio caso /odometry dà tutto il twist in body ma il modello usa solo
        # la parte angolare nel body, la lineare va ruotata in mondo
        self.current_vel[:] = [
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z,
        ]
        self.current_vel[:] = self.R @ self.current_vel[:]
        self.current_ang_vel[:] = [
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z,
        ]

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = msg.child_frame_id
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z
        t.transform.rotation = msg.pose.pose.orientation
        self.tf_broadcaster.sendTransform(t)

    def human_goal_callback(self, msg: PoseStamped):
        """
        Quando arriva un human_goal:
          - converti in [r, pan, tilt, roll, pitch, yaw]
          - attiva override fino a now + human_hold_s (param)
        """
        r = float(msg.pose.position.x)
        pan = float(msg.pose.position.y)
        tilt = float(msg.pose.position.z)
        q = msg.pose.orientation
        rpy = quat_to_RPY([q.w, q.x, q.y, q.z]).full().squeeze()
        roll, pitch, yaw = float(rpy[0]), float(rpy[1]), float(rpy[2])

        self.hgoal_ref = np.array([r, pan, tilt, roll, pitch, yaw], dtype=float)

        hold_human_ref = float(self.get_parameter('human_hold_ref').value)
        now = self.get_clock().now()
        # scadenza override
        from rclpy.duration import Duration
        self.hgoal_until = now + Duration(seconds=hold_human_ref)

        self.get_logger().info(
            f"human_goal ricevuto → override per {hold_human_ref:.2f}s | "
            f"ref: r={r:.2f}, pan={pan:.2f}, tilt={tilt:.2f}, rpy=({roll:.2f},{pitch:.2f},{yaw:.2f})"
        )

    # ==================== Configurazione e Solve ====================

    def configure_mpc(self):
        """Configura il solver MPC (chiamata allo START, non in peg_path_callback)."""
        # Pesi/limiti in linea con OCP (coerenza)
        D = 10; PANTILT = 2 * pi; V = 5; ANG = 1; ANG_DOT = pi/3
        ACC = 6; ACC_ANG = 200; JERK = 20; SNAP = 200
        U_F = 40; U_TAU = 0.3

        Q_pos = np.diag([10 / (D**2), 10 / (PANTILT**2), 10 / (PANTILT**2)])
        Q_vel = np.diag([5]*3) / V**2
        Q_rot = np.diag([1, 5, 5, 5]) / ANG**2
        Q_ang_dot = np.diag([3,3,4]) / ANG_DOT**2
        Q_acc = np.diag([2]*3) / ACC**2
        Q_acc_ang = np.diag([2, 2, 2]) / ACC_ANG**2
        Q_jerk = np.diag([5]*3) / JERK**2
        Q_snap = np.diag([5]*3) / SNAP**2

        R_f = np.diag([100]) / U_F**2
        R_tau = np.diag([10]*3) / U_TAU**2
        R = ca.diagcat(R_f, R_tau)
        Q = ca.diagcat(Q_pos, Q_vel, Q_rot, Q_ang_dot, Q_acc, Q_acc_ang, Q_jerk, Q_snap)

        W   = ca.diagcat(Q, R).full()
        W_e = 10 * Q.full()

        (self.ocp_solver,
         self.N_horiz, self.nx, self.nu,
         self.y_idx, self.ny, self.ny_e) = configure_mpc(
            model=self.model,
            x0=self.x0,
            p_obj=self.p_obj,
            rpy_obj=self.rpy_obj,
            Tf=self.Tp,
            ts=self.ts,
            W=W,
            W_e=W_e,
            ref=self.ref,
            final_ref=self.final_ref
        )

        # warm-start
        self.u_prev = [np.zeros(self.nu) for _ in range(self.N_horiz)]
        self.x_prev = [self.x0.copy()    for _ in range(self.N_horiz+1)]

        self.k = 0  # inizio timeline oggetto nella finestra attuale dell'MPC
        self.get_logger().info("MPC configurato")

        # Path predetto iniziale (da warm-start) per RViz
        self.publish_predicted_path_from_buffers()

    def solve_MPC(self, xk, online_ref):
        """
        Prepara parametri e yref sull'orizzonte e risolve l’MPC.
        Ritorna (u0, x_seq) con x_seq = [x0..xN].
        """
        set_initial_state(self.ocp_solver, xk)

        t0_idx = self.k
        M = len(self.p_obj)

        # aggiorna parametri+yref
        for i in range(self.N_horiz + 1):
            idx = min(t0_idx + i, M - 1)
            p_i   = self.p_obj[idx]
            rpy_i = self.rpy_obj[idx]
            param = np.concatenate([p_i, rpy_i, online_ref[3:]])  # [p_obj(3), rpy_obj(3), mut_rot_des(3)]
            self.ocp_solver.set(i, "p", param)
            if i < self.N_horiz:
                yref_i = build_yref_online(self.y_idx, online_ref)
                self.ocp_solver.set(i, "yref", yref_i)

        # terminal
        yref_e = build_yref_online(self.y_idx, online_ref)[:self.ny_e]
        self.ocp_solver.set(self.N_horiz, "yref", yref_e)

        # warm-start
        #for i in range(self.N_horiz):
        #    self.ocp_solver.set(i, "u", self.u_prev[i])
        #    self.ocp_solver.set(i, "x", self.x_prev[i])
        #self.ocp_solver.set(self.N_horiz, "x", self.x_prev[self.N_horiz])

        # solve
        status = self.ocp_solver.solve()
        if status != 0:
            self.get_logger().warn(f"MPC solve failed with status {status}")
            return None, None

        # estrai u0 e la sequenza degli stati
        u0 = self.ocp_solver.get(0, "u")
        x_seq = [self.ocp_solver.get(i, "x") for i in range(self.N_horiz + 1)]

        # shift warm-start
        for i in range(self.N_horiz - 1):
            self.u_prev[i] = self.ocp_solver.get(i + 1, "u")
            self.x_prev[i] = self.ocp_solver.get(i + 1, "x")
        if self.N_horiz > 1:
            self.u_prev[self.N_horiz - 1] = self.u_prev[self.N_horiz - 2].copy()
        else:
            self.u_prev[0] = self.ocp_solver.get(0, "u").copy()
        self.x_prev[self.N_horiz] = self.ocp_solver.get(self.N_horiz, "x")

        return u0, x_seq

    # ==================== Ciclo planner ====================

    def control_step(self):
        if not (self.mpc_ready and self.path_received):
            return

        # Stato iniziale xk (da odom; vel e ang vel non osservate → 0)
        roll, pitch, yaw = self.current_rpy
        q = RPY_to_quat(roll, pitch, yaw)
        xk = np.array([
            self.current_position[0], self.current_position[1], self.current_position[2],
            self.current_vel[0], self.current_vel[1], self.current_vel[2],
            float(q[0]), float(q[1]), float(q[2]), float(q[3]),
            self.current_ang_vel[0], self.current_ang_vel[1], self.current_ang_vel[2],
        ])

        # --- scelta del riferimento online ---
        now = self.get_clock().now()
        if self.hgoal_ref is not None and self.hgoal_until is not None and now < self.hgoal_until:
            online_ref = self.hgoal_ref
        else:
            online_ref = self.base_ref


        # Risoluzione MPC (planner)
        t0 = self.t_prev
        t1 = self.get_clock().now().nanoseconds * 1e-9

        u0, x_seq = self.solve_MPC(xk,online_ref)
        dt = (t1 - t0)
        print("tempo di chiamata control_step, iterazione ",self.k,": ", dt)
        if dt > 0.016:  # >80% del budget (0.02 s)
            self.get_logger().warn(f"MPC slow step: {dt*1000:.1f} ms")
            self.last_u0 = u0.copy() if u0 is not None else None  # solo per analisi/plot
        #if x_seq is None:
        #    # hold ultimo riferimento pubblicato (nessun cambiamento)
        #    self.get_logger().warn("MPC solve failed, holding last ref.")
        #    return

        # Pubblica stato seguente predetto (pose/twist) per il controllore esterno
        if x_seq is not None and len(x_seq) >= 2:
            self.publish_pose_and_twist(x_seq[1])

        if u0 is not None and len(u0) >= 2 :
            self.publish_optimal_wrench(u0)

        # (una volta) path completo iniziale per RViz
        if not self.mpc_path_published and x_seq is not None:
            self.publish_predicted_path(x_seq)
            self.mpc_path_published = True

        # avanza indice lungo la traiettoria dell’oggetto
        self.k = min(self.k + 1, len(self.p_obj) - 1)
        self.t_prev = t1

    # ==================== Pubblicazione (solo stato/visual) ====================

    def publish_pose_and_twist(self, x_vec):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "world"
        pose_msg.pose.position.x = float(x_vec[0])
        pose_msg.pose.position.y = float(x_vec[1])
        pose_msg.pose.position.z = float(x_vec[2])
        quat = x_vec[6:10]  # (w,x,y,z)
        pose_msg.pose.orientation.w = float(quat[0])
        pose_msg.pose.orientation.x = float(quat[1])
        pose_msg.pose.orientation.y = float(quat[2])
        pose_msg.pose.orientation.z = float(quat[3])
        self.single_pose_pub.publish(pose_msg)

        tw = TwistStamped()
        tw.header = pose_msg.header
        tw.twist.linear.x  = float(x_vec[3])
        tw.twist.linear.y  = float(x_vec[4])
        tw.twist.linear.z  = float(x_vec[5])
        tw.twist.angular.x = float(x_vec[10])
        tw.twist.angular.y = float(x_vec[11])
        tw.twist.angular.z = float(x_vec[12])
        self.single_twist_pub.publish(tw)

    def publish_optimal_wrench(self, u0) :
        #pubblicazione in terna body
        u_ff = u0
        wrench_msg = Wrench()
        wrench_msg.force.x = float(0.0)
        wrench_msg.force.y = float(0.0)
        wrench_msg.force.z = float(u_ff[0])
        wrench_msg.torque.x = float(u_ff[1])
        wrench_msg.torque.y = float(u_ff[2])
        wrench_msg.torque.z = float(u_ff[3])
        self.single_wrench_pub.publish(wrench_msg)

    def publish_predicted_path(self, x_seq):
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "world"
        for xi in x_seq:
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x = float(xi[0])
            ps.pose.position.y = float(xi[1])
            ps.pose.position.z = float(xi[2])
            quat = xi[6:10]
            ps.pose.orientation.w = float(quat[0])
            ps.pose.orientation.x = float(quat[1])
            ps.pose.orientation.y = float(quat[2])
            ps.pose.orientation.z = float(quat[3])
            path_msg.poses.append(ps)
        self.optimal_path_pub.publish(path_msg)

    def publish_predicted_path_from_buffers(self):
        if self.x_prev is None:
            return
        self.publish_predicted_path(self.x_prev)


def main(args=None):
    rclpy.init(args=args)
    node = MpcPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
