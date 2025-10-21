#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TwistStamped, Wrench
from nav_msgs.msg import Odometry
import numpy as np
from math import atan2
from drone_ocp_py.common import quat_to_R

def quat_to_yaw(qx, qy, qz, qw):
    siny_cosp = 2.0*(qw*qz + qx*qy)
    cosy_cosp = 1.0 - 2.0*(qy*qy + qz*qz)
    return atan2(siny_cosp, cosy_cosp)

def quat_to_rpy(qw, qx, qy, qz):
    # roll (x)
    sinr_cosp = 2*(qw*qx + qy*qz)
    cosr_cosp = 1 - 2*(qx*qx + qy*qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # pitch (y)
    sinp = 2*(qw*qy - qz*qx)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    # yaw (z)
    siny_cosp = 2*(qw*qz + qx*qy)
    cosy_cosp = 1 - 2*(qy*qy + qz*qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=float)

class Logger(Node):
    """
    Logger 'completo':
      - Parte a loggare dal primo /optimal_drone_pose ricevuto.
      - Rate limit a log_hz (timeline = /odometry).
      - Salva stato reale (pos, rpy, quat, vel, omega),
        riferimenti (pos, rpy, quat, vref, omegaref),
        wrench_cmd / wrench_ref e tempi.
    File .npz: t, pos, rpy, q, v, omega, p(=xyz+yaw), pref(=xyz+yaw),
               pref_pos, pref_rpy, pref_q, vref, omegaref,
               wrench_cmd, wrench_ref, t_ref
    """
    def __init__(self):
        super().__init__('logger')

        # --- parametri utente ---
        self.declare_parameter('save_path', '/tmp/pid_run.npz')
        self.declare_parameter('log_hz', 100.0)
        self.declare_parameter('save_ref_flag',True)

        self.save_path = self.get_parameter('save_path').value
        self.log_hz    = float(self.get_parameter('log_hz').value)
        self.log_dt    = 1.0 / max(self.log_hz, 1e-3)
        self.save_ref_flag = bool(self.get_parameter('save_ref_flag').value)

        # --- stato logging ---
        self.logging_enabled = False
        self.last_log_time = None

        # --- buffer (timeline = ODOM) ---
        self.t = []
        # stato reale
        self.pos = []     # (x,y,z)
        self.rpy = []     # (roll,pitch,yaw)
        self.q = []       # (w,x,y,z)
        self.v = []       # (vx,vy,vz)
        self.omega = []   # (wx,wy,wz)
        self.R = None   # actual rotation matrix (body -> world)

        # retro-compatibilità (come prima)
        self.p = []       # (x,y,z,yaw)

        # riferimenti (latch)
        self.pref_pos = []    # (x,y,z)
        self.pref_rpy = []    # (roll,pitch,yaw)
        self.pref_q = []      # (w,x,y,z)
        self.vref = []        # (vx,vy,vz)
        self.omegaref = []    # (wx,wy,wz)

        self.wrench_cmd = []  # (Fz, tx, ty, tz)
        self.wrench_ref = []  # (Fz, tx, ty, tz)
        self.t_ref = []       # timestamps arrivo pose ref (diagnostica)

        # --- ultimo riferimento noto (latch) ---
        self.last_pref_pos  = None
        self.last_pref_rpy  = None
        self.last_pref_q    = None
        self.last_vref      = None
        self.last_omegaref  = None
        self.last_w_cmd     = None
        self.last_w_ref     = None
        

        # --- subscribers ---
        #if self.save_ref_flag :
        self.create_subscription(PoseStamped,  '/optimal_drone_pose',  self.cb_ref_pose,   10)
        self.create_subscription(TwistStamped, '/optimal_drone_twist', self.cb_ref_twist,  10)
        self.create_subscription(Wrench,       '/optimal_wrench',      self.cb_wrench_ref, 10)

        self.create_subscription(Wrench,       '/wrench_cmd',          self.cb_wrench_cmd, 10)
        self.create_subscription(Odometry,     '/odometry',            self.cb_odom,       50)

        sim_time = self.get_parameter('use_sim_time').value if self.has_parameter('use_sim_time') else False
        self.get_logger().info(
            f'logger up | save_path={self.save_path} | log_hz={self.log_hz} | use_sim_time={sim_time}'
        )

    # ---------- utils ----------
    def now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ---------- callbacks ----------
    def cb_ref_pose(self, msg: PoseStamped):
        # posizione
        p = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=float)
        # orientamento
        qw = float(msg.pose.orientation.w)
        qx = float(msg.pose.orientation.x)
        qy = float(msg.pose.orientation.y)
        qz = float(msg.pose.orientation.z)
        rpy = quat_to_rpy(qw, qx, qy, qz)
        q = np.array([qw, qx, qy, qz], dtype=float)

        self.last_pref_pos = p
        self.last_pref_rpy = rpy
        self.last_pref_q   = q

        self.t_ref.append(self.now_sec())
        if not self.logging_enabled:
            self.logging_enabled = True
            self.last_log_time = None
            self.get_logger().info("Primo riferimento ricevuto → logging ON")

    def cb_ref_twist(self, msg: TwistStamped):
        self.last_vref = np.array([
            msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z
        ], dtype=float)
        self.last_omegaref = np.array([
            msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z
        ], dtype=float)

    def cb_wrench_ref(self, msg: Wrench):
        self.last_w_ref = [msg.force.z, msg.torque.x, msg.torque.y, msg.torque.z]

    def cb_wrench_cmd(self, msg: Wrench):
        self.last_w_cmd = [msg.force.z, msg.torque.x, msg.torque.y, msg.torque.z]

    def cb_odom(self, msg: Odometry):
        # se in offline (niente ref), il logging parte al primo odom (questo perché non legge riferimenti)
        #if not self.logging_enabled and not self.save_ref_flag:
        #    self.logging_enabled = True
        #    self.last_log_time = None
        #    self.get_logger().info("Primo ODOM ricevuto (offline) → logging ON")
        if not self.logging_enabled:
            return
        t_now = self.now_sec()
        if self.last_log_time is not None and (t_now - self.last_log_time < self.log_dt):
            return

        # stato reale
        px, py, pz = (msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z)
        qw = msg.pose.pose.orientation.w
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        roll, pitch, yaw = quat_to_rpy(qw, qx, qy, qz)
        q = [qw, qx, qy, qz]
        vx, vy, vz = (msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z)
        wx, wy, wz = (msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z)

        self.R = quat_to_R(q).full()
        # v : body (odometry)-> world
        [vx, vy, vz] = self.R @ [vx,vy,vz]

        self.t.append(t_now)
        self.pos.append([px, py, pz])
        self.rpy.append([roll, pitch, yaw])
        self.q.append([qw, qx, qy, qz])
        self.v.append([vx, vy, vz])
        self.omega.append([wx, wy, wz])

        # retro-compatibilità: p = [x,y,z,yaw]
        self.p.append([px, py, pz, yaw])

        # latch dei riferimenti
        self.pref_pos.append(self.last_pref_pos if self.last_pref_pos is not None else [np.nan]*3)
        self.pref_rpy.append(self.last_pref_rpy if self.last_pref_rpy is not None else [np.nan]*3)
        self.pref_q.append(self.last_pref_q if self.last_pref_q is not None else [np.nan]*4)

        self.vref.append(self.last_vref if self.last_vref is not None else [np.nan]*3)
        self.omegaref.append(self.last_omegaref if self.last_omegaref is not None else [np.nan]*3)

        self.wrench_cmd.append(self.last_w_cmd if self.last_w_cmd is not None else [np.nan]*4)
        self.wrench_ref.append(self.last_w_ref if self.last_w_ref is not None else [np.nan]*4)

        self.last_log_time = t_now

    def save(self):
        T = np.asarray(self.t)
        if T.size:
            T = T - T[0]

        out = dict(
            # timeline
            t=T,
            t_ref=np.asarray(self.t_ref),

            # stato reale
            pos=np.asarray(self.pos),
            rpy=np.asarray(self.rpy),
            q=np.asarray(self.q),
            v=np.asarray(self.v),
            omega=np.asarray(self.omega),

            # retro-compat (come prima)
            p=np.asarray(self.p),

            # riferimenti
            pref_pos=np.asarray(self.pref_pos) if len(self.pref_pos) else np.empty((0,3)),
            pref_rpy=np.asarray(self.pref_rpy) if len(self.pref_rpy) else np.empty((0,3)),
            pref_q=np.asarray(self.pref_q) if len(self.pref_q) else np.empty((0,4)),

            vref=np.asarray(self.vref) if len(self.vref) else np.empty((0,3)),
            omegaref=np.asarray(self.omegaref) if len(self.omegaref) else np.empty((0,3)),

            # wrench
            wrench_cmd=np.asarray(self.wrench_cmd) if len(self.wrench_cmd) else np.empty((0,4)),
            wrench_ref=np.asarray(self.wrench_ref) if len(self.wrench_ref) else np.empty((0,4)),
        )

        # --- PATCH: aggiungi 'pref' retro-compatibile (x,y,z,yaw) ---
        if len(self.pref_pos) and len(self.pref_rpy):
            try:
                pos = np.asarray(self.pref_pos)
                yaw = np.asarray(self.pref_rpy)[:, 2:3]  # solo colonna yaw
                out["pref"] = np.hstack([pos, yaw])
            except Exception as e:
                self.get_logger().warn(f"Impossibile costruire pref: {e}")
                out["pref"] = np.empty((0,4))
        else:
            out["pref"] = np.empty((0,4))

        np.savez(self.save_path, **out)

        hz_eff = (len(T) / max(T[-1], 1e-6)) if T.size else 0.0
        self.get_logger().info(f"saved {self.save_path} | N={len(T)} (~{hz_eff:.1f} Hz)")


def main(args=None):
    rclpy.init(args=args)
    node = Logger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.save()
        finally:
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
