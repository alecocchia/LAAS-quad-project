#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Wrench, TwistStamped
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Bool
import numpy as np
import casadi as ca
from casadi import pi as pi

from drone_ocp_py.drone_ocp_settings import (
    setup_model, setup_initial_conditions, configure_ocp,
    extract_trajectory_from_solver, get_state_variables
)
from drone_ocp_py.common import quat_to_RPY, RPY_to_R, RPY_to_quat

from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
import tf2_ros
from geometry_msgs.msg import TransformStamped


class OcpPlannerNode(Node):
    """
    Planner in modalità OPEN-LOOP:
    - Riceve /peg_path (traiettoria target), risolve OCP e pubblica /optimal_drone_path
    - Attende un "via" su /peg_pose, poi riproduce la sequenza pose x_opt(t) -> /optimal_drone_pose e di comandi u(t) -> /optimal_wrench
    - Nessun PID/feedback: comandi = feedforward dell'OCP
    """
    def __init__(self):
        super().__init__('ocp_planner_node')

        # --- Modello e condizioni iniziali ---
        self.model, self.model_rpy = setup_model()
        self.x0, self.x0_rpy = setup_initial_conditions()

        # --- Gestione tempi / orizzonte ---
        self.Tf = 20.0
        self.ts = 0.01
        self.ts_peg = 0.005
        self.N_horiz = int(self.Tf / self.ts)

        # --- Stato soluzione OCP ---
        self.ocp_solved = False
        self.x_opt_trajectory = None   # (N+1, nx)
        self.u_opt_trajectory = None   # (N,   nu)
        self.current_ocp_step = 0

        # --- Publisher "ready" e Path (entrambi latched) ---
        qos_latched_pub = QoSProfile(
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            depth=1
        )
        self.ready_publisher  = self.create_publisher(Bool, '/drone_planner_ready',  qos_latched_pub)
        self.optimal_path_pub = self.create_publisher(Path, '/optimal_drone_path', qos_latched_pub)

        # --- QoS per ricevere Path “latched” dal PEG ---
        qos_profile = QoSProfile(
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            depth=1
        )
        self.peg_path_subscription = self.create_subscription(
            Path, '/peg_path', self.peg_path_callback, qos_profile)

        # --- Odometry (solo per TF/visualizzazione) ---
        self.odom_subscription = self.create_subscription(
            Odometry, '/odometry', self.odom_callback, 1)

        # --- Pubblicazioni realtime ---
        self.wrench_commands_publisher = self.create_publisher(Wrench, '/optimal_wrench', 1)
        self.single_pose_pub = self.create_publisher(PoseStamped, '/optimal_drone_pose', 1)
        self.single_twist_pub = self.create_publisher(TwistStamped, '/optimal_drone_twist', 1)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # --- Start/trigger: prima posa del target su /peg_pose ---
        self.control_timer = None
        self.start_received = False
        self.start_subscription = self.create_subscription(
            PoseStamped, '/peg_pose', self.start_callback, 10)

        self.get_logger().info("OCP Planner Node (OPEN-LOOP) avviato. In attesa di /peg_path.")

        # --- Riferimenti ---
        radius = 2.0
        mut_pos_ref = np.array([radius, 0, 0])  # coordinate polari
        mut_rot_ref = np.array([0.0, 0.0, pi/2])  # rpy
        mut_pos_final_ref = np.array([radius, 0, 0])  # coordinate polari
        mut_rot_final_ref = np.array([0.0, 0.0, pi])  # rpy

        self.ref = np.concatenate([mut_pos_ref, mut_rot_ref])
        self.final_ref = np.concatenate([mut_pos_final_ref, mut_rot_final_ref])

        # Stato corrente (solo per TF)
        self.current_position = np.zeros(3)
        self.current_rpy = np.zeros(3)

        self._q_prev = None  # per continuità quaternion

    # -------------------- Callbacks --------------------

    def peg_path_callback(self, msg: Path):
        if self.ocp_solved:
            return

        p_obj_list, rpy_obj_list = [], []
        count = 0
        times_ratio = max(1, int(round(self.ts / self.ts_peg)))

        for pose_stamped in msg.poses:
            if count % times_ratio == 0:
                p = pose_stamped.pose.position
                q = pose_stamped.pose.orientation
                rpy = quat_to_RPY([q.w, q.x, q.y, q.z])  # w,x,y,z
                p_obj_list.append([p.x, p.y, p.z])
                rpy_obj_list.append(rpy)
            count += 1

        self.p_obj = np.array(p_obj_list)
        self.rpy_obj = np.squeeze(np.array(rpy_obj_list))

        self.solve_ocp()
        self.ocp_solved = True

        self.get_logger().info("OCP risolto. Invio segnale di 'ready' al peg planner per l'inizio della pubblicazione di pose.")
        ready_msg = Bool()
        ready_msg.data = True
        self.ready_publisher.publish(ready_msg)

    def start_callback(self, _msg: PoseStamped):
        if not self.start_received and self.ocp_solved:
            self.start_received = True
            self.get_logger().info("Start ricevuto. Avvio riproduzione feedforward OCP (open-loop).")
            self.control_timer = self.create_timer(self.ts, self.publish_next_point)
            self.destroy_subscription(self.start_subscription)

    def odom_callback(self, msg: Odometry):
        self.current_position[:] = [msg.pose.pose.position.x,
                                    msg.pose.pose.position.y,
                                    msg.pose.pose.position.z]
        self.current_rpy[:] = np.array(quat_to_RPY([msg.pose.pose.orientation.w,
                                           msg.pose.pose.orientation.x,
                                           msg.pose.pose.orientation.y,
                                           msg.pose.pose.orientation.z]).full()).squeeze()

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = msg.child_frame_id
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z
        t.transform.rotation = msg.pose.pose.orientation
        self.tf_broadcaster.sendTransform(t)

    # -------------------- OCP --------------------

    def solve_ocp(self):
        D = 10; PANTILT = 2 * pi; V = 5; ANG = 1; ANG_DOT = pi/3
        ACC = 6; ACC_ANG = 200; JERK = 20; SNAP = 200
        U_F = 40; U_TAU = 0.3

        Q_pos = np.diag([10 / (D**2), 10 / (PANTILT**2), 10 / (PANTILT**2)])
        Q_vel = np.diag([2]*3) / V**2
        Q_rot = np.diag([1, 5, 5, 5]) / ANG**2
        Q_ang_dot = np.diag([3,3,4]) / ANG_DOT**2
        Q_acc = np.diag([1]*3) / ACC**2
        Q_acc_ang = np.diag([2, 2, 2]) / ACC_ANG**2
        Q_jerk = np.diag([0.2]*3) / JERK**2
        Q_snap = np.diag([0.2]*3) / SNAP**2

        R_f = np.diag([10]) / U_F**2
        R_tau = np.diag([1]*3) / U_TAU**2
        R = ca.diagcat(R_f, R_tau)
        Q = ca.diagcat(Q_pos, Q_vel, Q_rot, Q_ang_dot, Q_acc, Q_acc_ang, Q_jerk, Q_snap)

        W   = ca.diagcat(Q, R).full()
        W_e = 10 * Q.full()

        self.ocp_solver, _, self.nx, self.nu = configure_ocp(
            model=self.model,
            x0=self.x0,
            p_obj=self.p_obj,
            rpy_obj=self.rpy_obj,
            Tf=self.Tf,
            ts=self.ts,
            W=W,
            W_e=W_e,
            ref=self.ref,
            final_ref=self.final_ref
        )

        status = self.ocp_solver.solve()
        if status != 0:
            self.get_logger().warn(f"OCP solve failed with status {status}")
            return
        else:
            self.get_logger().info(f"OCP solved in {self.ocp_solver.get_stats('time_tot'):.4f} s")

        simX, simU, simP, acc, jerk, snap = extract_trajectory_from_solver(
            self.ocp_solver, self.model, self.N_horiz, self.nx, self.nu
        )

        # Salva traiettoria ottima
        self.x_opt_trajectory = np.zeros((self.N_horiz + 1, self.nx))
        self.u_opt_trajectory = np.zeros((self.N_horiz, self.nu))

        optimal_path_msg = Path()
        optimal_path_msg.header.stamp = self.get_clock().now().to_msg()
        optimal_path_msg.header.frame_id = "world"

        for i in range(self.N_horiz + 1):
            self.x_opt_trajectory[i] = self.ocp_solver.get(i, "x")
            if i < self.N_horiz:
                self.u_opt_trajectory[i] = self.ocp_solver.get(i, "u")

            # Pose
            pose_stamped = PoseStamped()
            pose_stamped.header.stamp = self.get_clock().now().to_msg()
            pose_stamped.header.frame_id = "world"
            pose_stamped.pose.position.x = float(self.x_opt_trajectory[i, 0])
            pose_stamped.pose.position.y = float(self.x_opt_trajectory[i, 1])
            pose_stamped.pose.position.z = float(self.x_opt_trajectory[i, 2])
            quat = self.x_opt_trajectory[i, 6:10]  # (w,x,y,z)
            pose_stamped.pose.orientation.w = float(quat[0])
            pose_stamped.pose.orientation.x = float(quat[1])
            pose_stamped.pose.orientation.y = float(quat[2])
            pose_stamped.pose.orientation.z = float(quat[3])
            optimal_path_msg.poses.append(pose_stamped)

            # (Tolto) publish Twist qui per evitare doppia pubblicazione
            # Il Twist viene pubblicato step-by-step nel timer

        self.optimal_path_pub.publish(optimal_path_msg)
        self.get_logger().info("Traiettoria ottimale pubblicata su /optimal_drone_path.")

    def publish_next_point(self):
        if self.u_opt_trajectory is None or self.current_ocp_step >= self.N_horiz:
            i=self.N_horiz
            #u_ff = self.u_opt_trajectory[i-1]
            R_now = RPY_to_R(self.current_rpy[0], self.current_rpy[1], self.current_rpy[2])
            z_b0 = R_now[2,:] 
            u_ff = np.array([1.28*9.8, 0.0, 0.0, 0.0])/z_b0[2]  # [Fz, τx, τy, τz]
            if self.current_ocp_step == self.N_horiz:
                self.get_logger().info("Riproduzione comandi completata. Drone resterà in posizione e assetto finali")
        else :
            i = self.current_ocp_step
            u_ff = self.u_opt_trajectory[i]

        # Wrench
        wrench_msg = Wrench()
        wrench_msg.force.z = float(u_ff[0])
        wrench_msg.torque.x = float(u_ff[1])
        wrench_msg.torque.y = float(u_ff[2])
        wrench_msg.torque.z = float(u_ff[3])
        self.wrench_commands_publisher.publish(wrench_msg)

        # Pose
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "world"
        pose_msg.pose.position.x = float(self.x_opt_trajectory[i, 0])
        pose_msg.pose.position.y = float(self.x_opt_trajectory[i, 1])
        pose_msg.pose.position.z = float(self.x_opt_trajectory[i, 2])

        quat = self.x_opt_trajectory[i, 6:10].copy()

        pose_msg.pose.orientation.w = float(quat[0])
        pose_msg.pose.orientation.x = float(quat[1])
        pose_msg.pose.orientation.y = float(quat[2])
        pose_msg.pose.orientation.z = float(quat[3])
        self.single_pose_pub.publish(pose_msg)

        if i<self.N_horiz:
            # Twist al passo corrente (solo nel timer)
            tw = TwistStamped()
            tw.header.stamp = pose_msg.header.stamp
            tw.header.frame_id = "world"
            tw.twist.linear.x  = float(self.x_opt_trajectory[i, 3])
            tw.twist.linear.y  = float(self.x_opt_trajectory[i, 4])
            tw.twist.linear.z  = float(self.x_opt_trajectory[i, 5])
            tw.twist.angular.x = float(self.x_opt_trajectory[i, 10])
            tw.twist.angular.y = float(self.x_opt_trajectory[i, 11])
            tw.twist.angular.z = float(self.x_opt_trajectory[i, 12])
        else:
            #end of trajectory
            
            # Twist al passo corrente (solo nel timer)
            tw = TwistStamped()
            tw.header.stamp = pose_msg.header.stamp
            tw.header.frame_id = "world"
            tw.twist.linear.x  = 0.0
            tw.twist.linear.y  = 0.0
            tw.twist.linear.z  = 0.0
            tw.twist.angular.x = 0.0
            tw.twist.angular.y = 0.0
            tw.twist.angular.z = 0.0
        self.single_twist_pub.publish(tw)

        self.current_ocp_step += 1


def main(args=None):
    rclpy.init(args=args)
    node = OcpPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
