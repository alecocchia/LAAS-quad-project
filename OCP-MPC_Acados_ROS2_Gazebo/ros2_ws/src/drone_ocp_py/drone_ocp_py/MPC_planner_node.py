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
from std_msgs.msg import Bool, String, Float64MultiArray

import numpy as np
import casadi as ca
from casadi import pi as pi
from scipy.spatial.transform import Rotation
import time

from drone_ocp_py.drone_MPC_settings import (
    setup_model, setup_initial_conditions, configure_mpc, set_initial_state, build_yref_online, build_yref_terminal
)
from drone_ocp_py.common import quat_to_RPY, RPY_to_quat, R_to_RPY, RPY_to_R

import tf2_ros


class MpcPlannerNode(Node):
    def __init__(self):
        super().__init__('mpc_planner_node')

        # === Modello e condizioni iniziali (coerenti con OCP) ===
        # --- 1. Dichiarazione e Lettura Parametri da Launch File ---
        self.declare_parameter('mass', 1.28)
        self.declare_parameter('ixx', 0.023)
        self.declare_parameter('iyy', 0.023)
        self.declare_parameter('izz', 0.022)
        self.declare_parameter('cf', 8.0e-4) # Valori di default di sicurezza
        self.declare_parameter('ct', 1.0e-5)
        self.declare_parameter('start_x', 0.0)
        self.declare_parameter('start_y', 0.0)
        self.declare_parameter('start_z', 0.0)
        self.declare_parameter('start_roll', 0.0)
        self.declare_parameter('start_pitch', 0.0)
        self.declare_parameter('start_yaw', 0.0)
        self.declare_parameter('cam_x',0.0)     # camera pose in body frame
        self.declare_parameter('cam_y',0.0)
        self.declare_parameter('cam_z',0.0)
        self.declare_parameter('cam_roll',0.0)
        self.declare_parameter('cam_pitch',0.0)
        self.declare_parameter('cam_yaw',0.0)

        mass = self.get_parameter('mass').value
        ixx = self.get_parameter('ixx').value
        iyy = self.get_parameter('iyy').value
        izz = self.get_parameter('izz').value
        start_x = self.get_parameter('start_x').value
        start_y = self.get_parameter('start_y').value
        start_z = self.get_parameter('start_z').value
        start_roll = self.get_parameter('start_roll').value
        start_pitch = self.get_parameter('start_pitch').value
        start_yaw = self.get_parameter('start_yaw').value
        cam_x = self.get_parameter('cam_x').value
        cam_y = self.get_parameter('cam_y').value
        cam_z = self.get_parameter('cam_z').value
        cam_roll = self.get_parameter('cam_roll').value
        cam_pitch = self.get_parameter('cam_pitch').value
        cam_yaw = self.get_parameter('cam_yaw').value

        self.get_logger().info(f"Posizione iniziale drone: X={start_x}, Y={start_y}, Z={start_z}, R={start_roll}, P={start_pitch},Y={start_yaw}")
        self.get_logger().info(f"Posizione camera in body frame: X={cam_x}, Y={cam_y}, Z={cam_z}, R={cam_roll}, P={cam_pitch},Y={cam_yaw}")
        self.get_logger().info(f"Parametri SDF caricati: m={mass}, I=[{ixx}, {iyy}, {izz}]")

        # --- 2. Passaggio dei parametri al modello ---
        self.model, self.model_rpy = setup_model(mass, ixx, iyy, izz)
        
        self.x0, self.x0_rpy = setup_initial_conditions(start_x,start_y,start_z,start_roll,start_pitch,start_yaw)

        # === Tempo/Orizzonte (coerenti con OCP) ===
        self.Tf = 20.0
        num_campioni = 20
        self.ts = 0.02  # MPC va a 1/0.02 = 50 Hz
        self.Tp = num_campioni*self.ts # tempo di predizione (finestra MPC)
        self.ts_peg = 0.005
        self.N_horiz = int(self.Tf / self.ts)

        self.t_prev = 0.0

        # === Stato MPC / loop ===
        self.mpc_ready = False         # solver configurato
        self.path_received = False     # ricevuto peg_path
        self.is_peg_finished = False   # peg finisce la sua traiettoria -> inizio fase task
        self.start_received = False    # ricevuto /peg_pose (trigger)
        self.k = 0
        self.mpc_path_published = False

        self.u_prev = None
        self.x_prev = None
        self.last_u0 = None

        # === Dati target da /peg_path ===
        self.p_obj = None
        self.rpy_obj = None

        # === Offset della camera rispetto al cdm ===
        self.camera_offset = np.array([cam_x,cam_y,cam_z,cam_roll,cam_pitch,cam_yaw])


        # === Riferimenti mutual iniziali (come OCP) ===
        radius = 2.0
        mut_pos_ref = np.array([radius, 0.0, 0.0])   # [r, pan, tilt]
        mut_rot_ref = np.array([0.0, 0.0, pi])     # rpy
        mut_pos_final_ref = np.array([radius, 0, 0])
        mut_rot_final_ref = np.array([0.0, 0.0, pi])

        self.ref = np.concatenate([mut_pos_ref, mut_rot_ref])
        self.final_ref = np.concatenate([mut_pos_final_ref, mut_rot_final_ref])
        self.current_ref = self.ref.copy()  # aggiornabile via /human_goal

        # --- parametro: durata override umana (s) ---
        self.declare_parameter('human_hold_ref', 0.1)
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
        self.current_quat = np.zeros(4)
        self.current_raw_vel = np.zeros(3)
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
        
        self.peg_end_subscription = self.create_subscription(
            Bool, '/peg_path_finished', self.peg_path_finished_callback, qos_latched
        )

        # === Odometry (per TF/visual) ===
        self.odom_subscription = self.create_subscription(Odometry, '/odometry', self.odom_callback, 10)

        # === Pubblicazioni realtime (pose/twist predetti) ===
        self.single_pose_pub  = self.create_publisher(PoseStamped,  '/optimal_drone_pose',  1)
        self.single_twist_pub = self.create_publisher(TwistStamped, '/optimal_drone_twist', 1)
        self.single_wrench_pub = self.create_publisher(Wrench, wrench_topic_name, 1)
        self.tf_broadcaster   = tf2_ros.TransformBroadcaster(self)
        self.ref_pub = self.create_publisher(Float64MultiArray, '/online_ref', 1)

        # === Trigger di start ( come nell’OCP) ===
        self.control_timer = None
        self.start_subscription = self.create_subscription(PoseStamped, '/peg_pose', self.start_callback, 10)

        # === Goal umano per ref dinamico ===
        self.human_goal_sub = self.create_subscription(Float64MultiArray, 'human_goal', self.human_goal_callback, 10)

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

    def peg_path_finished_callback(self, msg: Bool):
        if msg.data is True:
            self.get_logger().info("Peg path is finished; starting task phase")
            self.is_peg_finished = True
        return

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
        # Assegna il quaternione per CasADi (w, x, y, z) leggendo direttamente dal messaggio
        self.current_quat[:] = [
            msg.pose.pose.orientation.w,
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z
        ]
        # Istanzia l'oggetto rotazione per poter ottenere altre rappresentazioni oltre al quaternione 
        # NB: il quaternione in rotation è x,y,z,w, quello del mio modello è  w,x,y,z
        rot_obj = Rotation.from_quat([msg.pose.pose.orientation.x,
                         msg.pose.pose.orientation.y,
                         msg.pose.pose.orientation.z,
                         msg.pose.pose.orientation.w])
        self.current_rpy[:] = rot_obj.as_euler('xyz')


        # Velocità dall'odometria:
        # NB: in molte configurazioni Gazebo dà twist in frame del child (body).
        # Siccome nel mio caso /odometry dà tutto il twist in body ma il modello usa solo
        # la parte angolare nel body, la lineare va ruotata in mondo
        self.current_raw_vel[:] = [
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z,
        ]
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

    def human_goal_callback(self, msg: Float64MultiArray):
        """
        Quando arriva un human_goal (Array di 3 elementi: r, pan, tilt):
          - lo espande a 6 elementi aggiungendo [0, 0, 0] per rpy 
            (l'assetto è gestito in automatico dal Visual Servoing)
          - attiva override fino a now + human_hold_s
        """
        if len(msg.data) >= 3:
            r = float(msg.data[0])
            pan = float(msg.data[1])
            tilt = float(msg.data[2])
            
            # online_ref richiede 6 elementi: [r, pan, tilt, roll, pitch, yaw]
            self.hgoal_ref = np.array([r, pan, tilt, 0.0, 0.0, 0.0], dtype=float)

            hold_human_ref = float(self.get_parameter('human_hold_ref').value)
            now = self.get_clock().now()
            from rclpy.duration import Duration
            self.hgoal_until = now + Duration(seconds=hold_human_ref)

            self.get_logger().info(
                f"human_goal ricevuto → override per {hold_human_ref:.2f}s | "
                f"ref: r={r:.2f}, pan={pan:.2f}, tilt={tilt:.2f}"
            )

    # ==================== Configurazione e Solve ====================

    def configure_mpc(self):
        """Configura il solver MPC (chiamata allo START)."""
        # Pesi/limiti in linea con OCP (coerenza)
        # Limiti operativi massimi (Normalizzazione)
        X = 10
        Y = 10
        Z = 10          # 10 metri max distanza
        #PANTILT = pi 
        V = 5.0            # 5 m/s max velocità operativa
        #ANG = 2 * pi       # angolo giro 
        QUAT = 1            # ora componenti qx,qy del quaternione
        ANG_DOT = 3.0      # 2 rad/s max
        ACC = 10.0         # 10 m/s^2 (~1g) di accelerazione lineare
        ACC_ANG = 11.0     # 11 rad/s^2 (Limite fisico: 0.15 Nm / 0.023 kgm^2)
        VISUAL = 2         # Y_max = r * tan (FoV_h/2), Y_max = r * tan (FoV_v/2)
        JERK = 20.0
        SNAP = 200.0
        U_F = 40.0         # 40 N max thrust
        U_TAU_XY = 0.25    # Max coppia Roll/Pitch
        U_TAU_Z = 0.15     # Max coppia Yaw

        # obiettivo primario
        PesoPos = 10.0
        # obiettivo visivo
        PesoVis = PesoPos / 2 
        #assetto
        PesoRot = PesoPos
        
        PesoVel = PesoPos / 4.0
        PesoAngVel = PesoRot / 2.0
        PesoAcc = PesoVel /10
        PesoAngAcc = PesoAngVel/10
        PesoJerk = PesoAcc / 10
        PesoSnap = PesoJerk

        PesoForce = PesoPos / 20
        PesoTorque = PesoForce*2

        Q_pos = np.diag([PesoPos,PesoPos]) / [X**2, Y**2]
        Q_visual = np.diag([PesoVis,PesoVis*1.5]) / VISUAL**2 # Y_c e Z_c
        Q_vel = np.diag([PesoVel, PesoVel, PesoVel]) / V**2
        Q_rot = np.diag([PesoRot, PesoRot]) / QUAT**2  
        
        Q_ang_dot = np.diag([PesoAngVel, PesoAngVel, PesoAngVel*0.5]) / ANG_DOT**2
        Q_acc = np.diag([PesoAcc, PesoAcc, PesoAcc*0.5]) / ACC**2
        Q_acc_ang = np.diag([PesoAngAcc, PesoAngAcc, PesoAngAcc*0.5]) / ACC_ANG**2
        Q_jerk = np.diag([PesoJerk, PesoJerk, PesoJerk*0.5]) / JERK**2
        Q_snap = np.diag([PesoSnap, PesoSnap, PesoSnap*0.5]) / SNAP**2

        R_f = np.diag([PesoForce]) / U_F**2
        R_tau = ca.diagcat(PesoTorque / U_TAU_XY**2, 
                           PesoTorque / U_TAU_XY**2, 
                           PesoTorque / U_TAU_Z**2)
        
        R = ca.diagcat(R_f, R_tau)
        Q = ca.diagcat(Q_pos, Q_visual, Q_vel, Q_rot, Q_ang_dot, Q_acc, Q_acc_ang, Q_jerk, Q_snap)

        W   = ca.diagcat(Q, R).full()
        W_e = 10* Q.full()

        p_init = self.p_obj[0]
        r_init, pan_init, tilt_init = self.ref[0], self.ref[1], self.ref[2]
        dummy_pos = np.array([
            p_init[0] + r_init * np.cos(tilt_init) * np.cos(pan_init),
            p_init[1] + r_init * np.cos(tilt_init) * np.sin(pan_init),
            p_init[2] + r_init * np.sin(tilt_init)
        ])
        dummy_ref = dummy_pos

        (self.ocp_solver,
         self.N_horiz, self.nx, self.nu,
         self.y_idx, self.ny, self.ny_e) = configure_mpc(
            model=self.model,
            x0=self.x0,
            camera_offset=self.camera_offset,
            p_obj=self.p_obj,
            rpy_obj=self.rpy_obj,
            Tf=self.Tp,
            ts=self.ts,
            W=W,
            W_e=W_e,
            ref=dummy_ref
        )

        # warm-start iniziale
        # --- INIZIALIZZAZIONE WRENCH DI HOVERING ---
        # Evita che l'MPC parta con i motori spenti (0 N) 
        mass = self.get_parameter('mass').value
        g0 = 9.81
        u_hover = np.array([mass * g0, 0.0, 0.0, 0.0])
        
        self.u_prev = [u_hover.copy() for _ in range(self.N_horiz)]        
        self.x_prev = [self.x0.copy()    for _ in range(self.N_horiz+1)]

        for i in range(self.N_horiz):
            self.ocp_solver.set(i, "u", self.u_prev[i])
            self.ocp_solver.set(i, "x", self.x_prev[i])
        self.ocp_solver.set(self.N_horiz, "x", self.x_prev[self.N_horiz])

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

        # --- PRE-CALCOLI FUORI DAL CICLO ---
        #mut_rot_des = online_ref[3:6]
        #R_mut_T = Rotation.from_euler('xyz', mut_rot_des).as_matrix().T

        r = online_ref[0]
        pan = online_ref[1]
        tilt = online_ref[2]
        
        offset_x = r * np.cos(tilt) * np.cos(pan)
        offset_y = r * np.cos(tilt) * np.sin(pan)
        offset_z = r * np.sin(tilt)

        #q_current = xk[6:10]    # Quaternione attuale del drone per il Filtro Emisfero
        # -----------------------------------

        # aggiorna parametri+yref
        for i in range(self.N_horiz + 1):
            idx = min(t0_idx + i, M - 1)
            p_i   = self.p_obj[idx]
            #rpy_i = self.rpy_obj[idx]
            
            # Calcolo target di orientamento tramite prodotto di matrici
            #R_obj = Rotation.from_euler('xyz', rpy_i).as_matrix()
            #R_target = R_obj @ R_mut_T
            
            # RPY
            #rpy_target = Rotation.from_matrix(R_target).as_euler('xyz')
            #rp_target = np.array([0.0,0.0])  # Roll e pitch target forzati a 0
        
            # --- GESTIONE WRAP-AROUND DELLO YAW --- (NON FUNZIONA BENE, IL DRONE GIRA COME UNA TROTTOLA)
            #current_yaw = quat_to_RPY(q_current)[2].full().item() 
            #yaw_error = rpy_target[2] - current_yaw
            #yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi
            #rpy_target[2] = current_yaw + yaw_error
            
            # Conversione in quaternione target
            #q_target = rp_target.copy() #il target dei quaternioni è dato da qx e qy a 0 (suppongo drone orizzontale)
            
            # Filtro Emisfero
            #if np.dot(q_current, q_target) < 0:
            #    q_target = -q_target
            
            # Conversione da coordinate sferiche (r, pan, tilt) a coordinate cartesiane assolute (X, Y, Z)
            pos_target = np.array([
                p_i[0] + offset_x,
                p_i[1] + offset_y,
                p_i[2] + offset_z
            ])
            
            ref_vec = pos_target

            param = p_i     #pos dell'oggetto all'istante i  
            self.ocp_solver.set(i, "p", param)
            
            if i < self.N_horiz:
                yref_i = build_yref_online(self.y_idx, ref_vec)
                self.ocp_solver.set(i, "yref", yref_i)
            elif i == self.N_horiz:
                # terminal
                yref_e = build_yref_online(self.y_idx, ref_vec)[:self.ny_e]
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
            u0=self.u_prev[0].copy()
            x_seq = [self.x_prev[i].copy() for i in range(self.N_horiz + 1)]
            # shift warm-start
            for i in range(self.N_horiz - 1):
                self.u_prev[i] = self.u_prev[i+1].copy()
                self.x_prev[i] = self.x_prev[i+1].copy()
            #duplicazione ultimo comando per non avere buchi
            if self.N_horiz > 1:
                self.u_prev[self.N_horiz - 1] = self.u_prev[self.N_horiz - 2].copy()
            else:
                self.u_prev[0] = self.u_prev[-1].copy() 
            self.x_prev[self.N_horiz] = self.x_prev[-1].copy()
            return u0, x_seq

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
        self.R = Rotation.from_euler('xyz',self.current_rpy).as_matrix()
        self.current_vel[:] = self.R @ self.current_raw_vel[:]

        xk = np.array([
            self.current_position[0], self.current_position[1], self.current_position[2],
            self.current_vel[0], self.current_vel[1], self.current_vel[2],
            self.current_quat[0], self.current_quat[1], self.current_quat[2], self.current_quat[3],
            self.current_ang_vel[0], self.current_ang_vel[1], self.current_ang_vel[2],
        ])

        # --- scelta del riferimento online ---
        now = self.get_clock().now()
        if self.hgoal_ref is not None and self.hgoal_until is not None and now < self.hgoal_until:  # Fase di input dinamico dell'uomo
            online_ref = self.hgoal_ref
        else:
            if self.is_peg_finished is True:
                online_ref = self.final_ref # Fase 2: il peg è fermo in posizione finale --> fase di task
            else:
                online_ref = self.base_ref  # Fase 1: il peg si sta muovendo ed il task ancora deve cominciare
        
        # --- PUBBLICAZIONE RIFERIMENTO ONLINE PER IL LOGGER ---
        ref_msg = Float64MultiArray()
        ref_msg.data = [float(x) for x in online_ref]
        self.ref_pub.publish(ref_msg)


        # Risoluzione MPC (planner)
        t1 = self.get_clock().now().nanoseconds * 1e-9 # Aggiorno t1 per la logica di t_prev

        t_start = time.perf_counter()
        u0, x_seq = self.solve_MPC(xk,online_ref)
        t_end = time.perf_counter()
        
        dt = t_end - t_start
        #print("tempo di chiamata control_step, iterazione ",self.k,": ", dt)
        
        if dt > 1 * self.ts:  # >80% del budget (0.02 s)
            self.get_logger().warn(f"MPC slow step (> 100% ts): {dt*1000:.1f} ms")
            self.last_u0 = u0.copy() if u0 is not None else None  # solo per analisi/plot
        #if x_seq is None:
        #    # hold ultimo riferimento pubblicato (nessun cambiamento)
        #    self.get_logger().warn("MPC solve failed, holding last ref.")
        #    return

        # Pubblica stato seguente predetto (pose/twist) per eventuale controllore esterno
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