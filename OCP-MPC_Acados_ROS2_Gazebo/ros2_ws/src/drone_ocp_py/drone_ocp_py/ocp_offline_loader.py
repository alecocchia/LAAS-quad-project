#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from geometry_msgs.msg import PoseStamped, TwistStamped, Wrench
from nav_msgs.msg import Path
from std_msgs.msg import Bool
import numpy as np

def rpy_to_quat(roll, pitch, yaw):
    """Ritorna (w,x,y,z)."""
    cr = np.cos(0.5 * roll); sr = np.sin(0.5 * roll)
    cp = np.cos(0.5 * pitch); sp = np.sin(0.5 * pitch)
    cy = np.cos(0.5 * yaw); sy = np.sin(0.5 * yaw)
    qw = cr*cp*cy + sr*sp*sy
    qx = sr*cp*cy - cr*sp*sy
    qy = cr*sp*cy + sr*cp*sy
    qz = cr*cp*sy - sr*sp*cy
    return float(qw), float(qx), float(qy), float(qz)

def yaw_to_quat(yaw):
    """Convenzione roll=pitch=0, ritorna (w,x,y,z)."""
    return rpy_to_quat(0.0, 0.0, float(yaw))

class OcpOfflineLoader(Node):
    """
    Replay offline delle reference salvate dal logger:
      - Pubblica /optimal_drone_path (Transient Local) con keep-alive.
      - Streamma /optimal_drone_pose, /optimal_drone_twist (lineare+angolare) e opzionalmente /optimal_wrench.
    Il file .npz può contenere:
      Nuovo:   pref_pos(N,3), pref_rpy(N,3), pref_q(N,4 wxyz), vref(N,3), omegaref(N,3)
      Vecchio: pref(N,4 [x,y,z,yaw]), vref(N,3)
      Wrench:  wrench_ref(N,4)
      Tempo:   t(N) relativo (inizio a 0). Se assente, deduce dt da rate_hz.
    """
    def __init__(self):
        super().__init__('ocp_offline_loader')

        # --- Parametri ---
        self.declare_parameter('log_file', '/tmp/pid_run.npz')
        self.declare_parameter('frame_id', 'world')
        self.declare_parameter('rate_hz', 0.0)               # 0 => usa t del log
        self.declare_parameter('publish_wrench', True)
        self.declare_parameter('keep_alive_hz', 1.0)         # repubblica Path per RViz

        log_path       = self.get_parameter('log_file').value
        self.frame_id  = self.get_parameter('frame_id').value
        rate_hz        = float(self.get_parameter('rate_hz').value)
        self.pub_wrench= bool(self.get_parameter('publish_wrench').value)
        keep_alive_hz  = float(self.get_parameter('keep_alive_hz').value)


        # --- QoS ---
        qos_latched = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self.path_pub   = self.create_publisher(Path,         '/optimal_drone_path', qos_latched)
        self.pose_pub   = self.create_publisher(PoseStamped,  '/optimal_drone_pose', 1)
        self.twist_pub  = self.create_publisher(TwistStamped, '/optimal_drone_twist', 1)
        self.wrench_pub = self.create_publisher(Wrench,       '/optimal_wrench', 1)
        self.ready_pub  = self.create_publisher(Bool,         '/OCP_planner_ready', qos_latched)

        # --- Carica dati ---
        try:
            data = np.load(log_path)
        except Exception as e:
            self.get_logger().error(f"Impossibile aprire {log_path}: {e}")
            raise

        # Tempo
        self.t = data['t'] if 't' in data.files else None

        # Riferimenti posizione/orientamento
        # Precedenza: pref_q -> pref_rpy -> pref_pos+pref[:,3] -> pref (xyz+yaw)
        self.pref_pos = data['pref_pos'] if 'pref_pos' in data.files else None
        self.pref_rpy = data['pref_rpy'] if 'pref_rpy' in data.files else None
        self.pref_q   = data['pref_q']   if 'pref_q'   in data.files else None
        self.pref_xyzyaw = data['pref']  if 'pref'     in data.files else None

        # Velocità riferimento (lineare/angolare)
        self.vref      = data['vref']      if 'vref'      in data.files else None
        self.omegaref  = data['omegaref']  if 'omegaref'  in data.files else None

        # Wrench riferimento
        self.wref = data['wrench_ref'] if 'wrench_ref' in data.files else None

        # --- Costruisci sequenza di campioni coerente ---
        # deduci N e crea pos(N,3), quat(N,4), yaw(N), vref(N,3), omegaref(N,3)
        if self.pref_q is not None:
            N = self.pref_q.shape[0]
        elif self.pref_rpy is not None:
            N = self.pref_rpy.shape[0]
        elif self.pref_pos is not None:
            N = self.pref_pos.shape[0]
        elif self.pref_xyzyaw is not None:
            N = self.pref_xyzyaw.shape[0]
        else:
            self.get_logger().error("Nessun riferimento posizione/orientamento trovato nel log.")
            raise RuntimeError("Log privo di pref_*")

        # pos
        if self.pref_pos is not None:
            pos = np.asarray(self.pref_pos, dtype=float)
        elif self.pref_xyzyaw is not None:
            pos = np.asarray(self.pref_xyzyaw[:, :3], dtype=float)
        else:
            self.get_logger().error("Manca pref_pos (o pref[:,0:3]) nel log.")
            raise RuntimeError("pos ref mancante")

        # quat (w,x,y,z)
        if self.pref_q is not None:
            quat = np.asarray(self.pref_q, dtype=float)
        elif self.pref_rpy is not None:
            rpy = np.asarray(self.pref_rpy, dtype=float)
            quat = np.zeros((N, 4), dtype=float)
            for i in range(N):
                qw, qx, qy, qz = rpy_to_quat(rpy[i,0], rpy[i,1], rpy[i,2])
                quat[i] = [qw, qx, qy, qz]
        elif self.pref_xyzyaw is not None:
            yaw = np.asarray(self.pref_xyzyaw[:, 3], dtype=float)
            quat = np.zeros((N, 4), dtype=float)
            for i in range(N):
                quat[i] = yaw_to_quat(yaw[i])
        else:
            self.get_logger().error("Manca pref_q/pref_rpy/pref[:,3] per costruire il quaternione.")
            raise RuntimeError("orient ref mancante")

        # vref (lineare)
        vref = np.zeros((N, 3), dtype=float)
        if self.vref is not None and self.vref.shape[0] >= N:
            vref[:] = np.asarray(self.vref[:N], dtype=float)

        # omegaref (angolare): preferisci dal log; altrimenti calcola yawdot come fallback
        omegaref = np.zeros((N, 3), dtype=float)
        if self.omegaref is not None and self.omegaref.shape[0] >= N:
            omegaref[:] = np.asarray(self.omegaref[:N], dtype=float)
        else:
            # fallback da yaw (se disponibile) → solo z
            if self.pref_rpy is not None:
                yaw = np.asarray(self.pref_rpy[:, 2], dtype=float)
            elif self.pref_xyzyaw is not None:
                yaw = np.asarray(self.pref_xyzyaw[:, 3], dtype=float)
            else:
                yaw = np.unwrap(np.arctan2(2*(quat[:,0]*quat[:,3] + quat[:,1]*quat[:,2]),
                                           1 - 2*(quat[:,2]**2 + quat[:,3]**2)))
            yaw_unw = np.unwrap(yaw)
            # dt provvisorio per yawdot (sistemiamo subito dopo)
            if rate_hz > 0:
                dt_tmp = 1.0 / rate_hz
            elif (self.t is not None) and (len(self.t) > 1):
                dt_tmp = float(np.median(np.diff(self.t)))
            else:
                dt_tmp = 0.01
            yawdot = np.gradient(yaw_unw, dt_tmp)
            omegaref[:, 2] = yawdot

        # wrench ref
        wref = None
        if self.wref is not None and self.wref.shape[0] >= N:
            wref = np.asarray(self.wref[:N], dtype=float)

        # --- Timing (dt) ---
        if rate_hz > 0:
            self.dt = 1.0 / rate_hz
        else:
            self.dt = float(np.median(np.diff(self.t))) if (self.t is not None and len(self.t) > 1) else 0.01

        self.get_logger().info(f"Caricati {N} campioni da {log_path} | dt≈{self.dt:.4f} s")

        # --- Salva buffers pronti all’uso ---
        self.N = N
        self.pos  = pos
        self.quat = quat
        self.vref = vref
        self.omegaref = omegaref
        self.wref = wref

        # --- Pubblica Path (come l'online) + keep-alive ---
        self._publish_path_and_arm_keepalive()

        # --- Invia segnale ready (latched) ---
        self.ready_pub.publish(Bool(data=True))
        self.get_logger().info("OFFLINE: inviato (latched) /OCP_planner_ready=True")

        # --- Timer replay ---
        self.i = 0
        self.timer = self.create_timer(self.dt, self._tick)

        # --- Keep-alive per Path (RViz) ---
        if keep_alive_hz > 1e-6:
            self._path_timer = self.create_timer(1.0/keep_alive_hz, self._republish_path)

    # ---------- Path ----------
    def _publish_path_and_arm_keepalive(self):
        path = Path()
        path.header.frame_id = self.frame_id
        path.header.stamp = self.get_clock().now().to_msg()

        for i in range(self.N):
            ps = PoseStamped()
            ps.header.frame_id = self.frame_id
            ps.header.stamp = path.header.stamp  # uniforme (RViz-friendly)
            ps.pose.position.x = float(self.pos[i, 0])
            ps.pose.position.y = float(self.pos[i, 1])
            ps.pose.position.z = float(self.pos[i, 2])
            qw, qx, qy, qz = self.quat[i]
            ps.pose.orientation.w = float(qw)
            ps.pose.orientation.x = float(qx)
            ps.pose.orientation.y = float(qy)
            ps.pose.orientation.z = float(qz)
            path.poses.append(ps)

        self.path_pub.publish(path)
        self._path_msg = path
        self.get_logger().info(f"Path pubblicato con {len(path.poses)} pose.")

    def _republish_path(self):
        # Aggiorna solo lo stamp dell'header (come fa l'online quando crea i msg)
        self._path_msg.header.stamp = self.get_clock().now().to_msg()
        self.path_pub.publish(self._path_msg)

    # ---------- Replay step ----------
    def _tick(self):
        i = self.i
        if i >= self.N:
            self.get_logger().info("Fine riproduzione offline.")
            self.timer.cancel()
            # keep-alive dell’ultima pose per un po’ (evita “cadute” dei follower)
            last_pose = PoseStamped()
            last_pose.header.frame_id = self.frame_id
            last_pose.header.stamp = self.get_clock().now().to_msg()
            last_pose.pose.position.x = float(self.pos[-1,0])
            last_pose.pose.position.y = float(self.pos[-1,1])
            last_pose.pose.position.z = float(self.pos[-1,2])
            qw, qx, qy, qz = self.quat[-1]
            last_pose.pose.orientation.w = float(qw)
            last_pose.pose.orientation.x = float(qx)
            last_pose.pose.orientation.y = float(qy)
            last_pose.pose.orientation.z = float(qz)

            # 30 Hz per ~3 s
            for _ in range(90):
                last_pose.header.stamp = self.get_clock().now().to_msg()
                self.pose_pub.publish(last_pose)
                rclpy.sleep(0.033) if hasattr(rclpy, 'sleep') else None
            return

        # Pose
        pose = PoseStamped()
        pose.header.frame_id = self.frame_id
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(self.pos[i, 0])
        pose.pose.position.y = float(self.pos[i, 1])
        pose.pose.position.z = float(self.pos[i, 2])
        qw, qx, qy, qz = self.quat[i]
        pose.pose.orientation.w = float(qw)
        pose.pose.orientation.x = float(qx)
        pose.pose.orientation.y = float(qy)
        pose.pose.orientation.z = float(qz)
        self.pose_pub.publish(pose)

        # Twist (lineare + angolare)
        tw = TwistStamped()
        tw.header = pose.header
        tw.twist.linear.x  = float(self.vref[i, 0]) if np.isfinite(self.vref[i]).all() else 0.0
        tw.twist.linear.y  = float(self.vref[i, 1]) if np.isfinite(self.vref[i]).all() else 0.0
        tw.twist.linear.z  = float(self.vref[i, 2]) if np.isfinite(self.vref[i]).all() else 0.0
        tw.twist.angular.x = float(self.omegaref[i, 0]) if np.isfinite(self.omegaref[i]).all() else 0.0
        tw.twist.angular.y = float(self.omegaref[i, 1]) if np.isfinite(self.omegaref[i]).all() else 0.0
        tw.twist.angular.z = float(self.omegaref[i, 2]) if np.isfinite(self.omegaref[i]).all() else 0.0
        self.twist_pub.publish(tw)

        # Wrench (se disponibile e richiesto)
        if self.pub_wrench and (self.wref is not None) and (self.wref.shape[0] > i):
            if np.isfinite(self.wref[i]).all():
                wmsg = Wrench()
                wmsg.force.z  = float(self.wref[i, 0])
                wmsg.torque.x = float(self.wref[i, 1])
                wmsg.torque.y = float(self.wref[i, 2])
                wmsg.torque.z = float(self.wref[i, 3])
                self.wrench_pub.publish(wmsg)

        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    node = OcpOfflineLoader()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
