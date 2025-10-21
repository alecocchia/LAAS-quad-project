#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Path

def rpy_to_quat(roll, pitch, yaw):
    cr = math.cos(0.5*roll);  sr = math.sin(0.5*roll)
    cp = math.cos(0.5*pitch); sp = math.sin(0.5*pitch)
    cy = math.cos(0.5*yaw);   sy = math.sin(0.5*yaw)
    qw = cr*cp*cy + sr*sp*sy
    qx = sr*cp*cy - cr*sp*sy
    qy = cr*sp*cy + sr*cp*sy
    qz = cr*cp*sy - sr*cp*cy
    return qw, qx, qy, qz

class PlannerProva(Node):
    """
    Generatore di riferimenti sinusoidali selezionabile:
      - target ∈ {x, y, z, yaw}
      - q(t) = q0 + A * sin(2π f t)
      - dq/dt = 2π f A * cos(2π f t) (lineare per x/y/z, angolare z per yaw)
    Pubblica:
      /optimal_drone_pose  (PoseStamped)
      /optimal_drone_twist (TwistStamped)
      /optimal_drone_path  (Path, latched per RViz/diagnostica)
      /OCP_planner_ready   (Bool,  latched)
    """
    def __init__(self):
        super().__init__('planner_prova')

        # --- Parametri ---
        self.declare_parameter('frame_id', 'world')
        self.declare_parameter('rate_hz', 100.0)       # Hz di pubblicazione pose/twist
        self.declare_parameter('x0', 0.0)
        self.declare_parameter('y0', 0.0)
        self.declare_parameter('z0', 5.0)
        self.declare_parameter('yaw', 0.0)             # rad
        self.declare_parameter('amp', 2.0)             # ampiezza sinusoide
        self.declare_parameter('freq_hz', 0.2)         # frequenza (Hz)
        self.declare_parameter('sinusoid_target', 'x') # {x,y,z,yaw}
        self.declare_parameter('publish_path', True)   # Path latched per RViz
        self.declare_parameter('path_seconds', 100.0)  # durata Path

        self.frame_id     = self.get_parameter('frame_id').value
        self.rate_hz      = float(self.get_parameter('rate_hz').value)
        self.dt           = 1.0 / max(self.rate_hz, 1e-6)
        self.x0           = float(self.get_parameter('x0').value)
        self.y0           = float(self.get_parameter('y0').value)
        self.z0           = float(self.get_parameter('z0').value)
        self.yaw0         = float(self.get_parameter('yaw').value)
        self.amp          = float(self.get_parameter('amp').value)
        self.freq_hz      = float(self.get_parameter('freq_hz').value)
        self.target       = str(self.get_parameter('sinusoid_target').value).lower()
        self.publish_path = bool(self.get_parameter('publish_path').value)
        self.path_seconds = float(self.get_parameter('path_seconds').value)

        # Validazione semplice del target
        if self.target not in ('x', 'y', 'z', 'yaw'):
            self.get_logger().warn(f"sinusoid_target='{self.target}' non valido. Uso 'z'.")
            self.target = 'z'

        # --- Publisher (QoS) ---
        qos_latched = QoSProfile(depth=1,
                                 durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                                 reliability=QoSReliabilityPolicy.RELIABLE)
        self.path_pub  = self.create_publisher(Path, '/optimal_drone_path', qos_latched)
        self.ready_pub = self.create_publisher(Bool, '/OCP_planner_ready', qos_latched)

        self.pose_pub  = self.create_publisher(PoseStamped,  '/optimal_drone_pose',  1)
        self.twist_pub = self.create_publisher(TwistStamped, '/optimal_drone_twist', 1)

        # --- Stato interno tempo continuo (secondi) ---
        self.t = 0.0

        # --- Pubblica Path latched (opzionale) ---
        if self.publish_path:
            self._publish_path_latched()

        # --- Pubblica READY latched ---
        self.ready_pub.publish(Bool(data=True))
        self.get_logger().info("PlannerProva READY: /OCP_planner_ready=True (latched)")

        # --- Timer di streaming pose/twist ---
        self.timer = self.create_timer(self.dt, self._tick)
        self.get_logger().info(
            f"PlannerProva avviato | target={self.target}, q0(x,y,z,yaw)=({self.x0},{self.y0},{self.z0},{self.yaw0}), "
            f"A={self.amp}, f={self.freq_hz} Hz, rate={self.rate_hz} Hz"
        )

    # --------- Path (diagnostica/RViz) ----------
    def _publish_path_latched(self):
        path = Path()
        path.header.frame_id = self.frame_id
        path.header.stamp = self.get_clock().now().to_msg()

        N = max(1, int(self.path_seconds / self.dt))
        w = 2.0 * math.pi * self.freq_hz

        for k in range(N):
            # Riferimento costante alla fine
            if k>N/10 :
                ps = PoseStamped()
                ps.header.frame_id = self.frame_id
                ps.header.stamp = path.header.stamp  # timestamp uniforme (RViz-friendly)
                ps.pose.position.x = float(self.x0)
                ps.pose.position.y = float(self.y0)
                ps.pose.position.z = float(self.z0)
                ps.pose.orientation.w = float(1)
                ps.pose.orientation.x = float(0)
                ps.pose.orientation.y = float(0)
                ps.pose.orientation.z = float(0)
                path.poses.append(ps)
            else:   
                tk = k * self.dt

                # Valori base
                x = self.x0
                y = self.y0
                z = self.z0
                yaw = self.yaw0

                # Applica sinusoide al target
                s = self.amp * math.sin(w * tk)
                if self.target == 'x':
                    x = self.x0 + s
                elif self.target == 'y':
                    y = self.y0 + s
                elif self.target == 'z':
                    z = self.z0 + s
                elif self.target == 'yaw':
                    yaw = self.yaw0 + s
                    z = self.z0 + s

                qw, qx, qy, qz = rpy_to_quat(0.0, 0.0, yaw)

                ps = PoseStamped()
                ps.header.frame_id = self.frame_id
                ps.header.stamp = path.header.stamp  # timestamp uniforme (RViz-friendly)
                ps.pose.position.x = float(x)
                ps.pose.position.y = float(y)
                ps.pose.position.z = float(z)
                ps.pose.orientation.w = float(qw)
                ps.pose.orientation.x = float(qx)
                ps.pose.orientation.y = float(qy)
                ps.pose.orientation.z = float(qz)
                path.poses.append(ps)

        self.path_pub.publish(path)
        self.get_logger().info(f"Path latched pubblicato: {len(path.poses)} pose (durata ~{self.path_seconds:.1f}s)")

    # --------- Streaming pose/twist ----------
    def _tick(self):
        # Valori base
        x = self.x0
        y = self.y0
        z = self.z0
        yaw = self.yaw0
        # Derivate (per twist)
        vx = 0.0
        vy = 0.0
        vz = 0.0
        wz = 0.0

        if self.t > self.path_seconds/10 :
            # Riferimento costante alla fine
            # Pose
            pose = PoseStamped()
            pose.header.frame_id = self.frame_id
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = float(z)
            qw, qx, qy, qz = rpy_to_quat(0.0, 0.0, yaw+math.pi)
            pose.pose.orientation.w = float(qw)
            pose.pose.orientation.x = float(qx)
            pose.pose.orientation.y = float(qy)
            pose.pose.orientation.z = float(qz)
            self.pose_pub.publish(pose)
    
            # Twist coerente
            tw = TwistStamped()
            tw.header = pose.header
            tw.twist.linear.x  = 0.0
            tw.twist.linear.y  = 0.0
            tw.twist.linear.z  = 0.0
            tw.twist.angular.x = 0.0
            tw.twist.angular.y = 0.0
            tw.twist.angular.z = 0.0
            self.twist_pub.publish(tw)
        else :
            w = 2.0 * math.pi * self.freq_hz
    
            # Applicazione della sinusoide al target
            s  = self.amp * math.sin(w * self.t)
            ds = self.amp * w * math.cos(w * self.t)
    
            if self.target == 'x':
                x = self.x0 + s
                vx = ds
            elif self.target == 'y':
                y = self.y0 + s
                vy = ds
            elif self.target == 'z':
                z = self.z0 + s
                vz = ds
            elif self.target == 'yaw':
                yaw = self.yaw0 + s
                wz = ds
            
            # Eventualmente applicare sinusoide allo yaw a prescindere dalla variabile selezionata
            #yaw = self.yaw0 + s

            # Pose
            pose = PoseStamped()
            pose.header.frame_id = self.frame_id
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = float(z)
            qw, qx, qy, qz = rpy_to_quat(0.0, 0.0, yaw)
            pose.pose.orientation.w = float(qw)
            pose.pose.orientation.x = float(qx)
            pose.pose.orientation.y = float(qy)
            pose.pose.orientation.z = float(qz)
            self.pose_pub.publish(pose)
    
            # Twist coerente
            tw = TwistStamped()
            tw.header = pose.header
            tw.twist.linear.x  = float(vx)
            tw.twist.linear.y  = float(vy)
            tw.twist.linear.z  = float(vz)
            tw.twist.angular.x = 0.0
            tw.twist.angular.y = 0.0
            tw.twist.angular.z = float(wz)
            self.twist_pub.publish(tw)

        self.t += self.dt

def main(args=None):
    rclpy.init(args=args)
    node = PlannerProva()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
