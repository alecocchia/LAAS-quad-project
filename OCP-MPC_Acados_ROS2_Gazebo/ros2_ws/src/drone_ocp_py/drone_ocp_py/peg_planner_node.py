import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Bool
import numpy as np
from drone_ocp_py.common import RPY_to_quat
from drone_ocp_py.planner import generate_trapezoidal_trajectory
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

class PegPlannerNode(Node):
    def __init__(self):
        super().__init__('peg_planner_node')

        self.t0 = 0
        self.Tf = 20.0
        self.ts = 0.005

        p_obj_in = np.array([2 , 2 , 0.5])
        p_obj_f  = np.array([5, 5, 10.5])
        rot_obj_in = np.array([0,0,0])
        rot_obj_f  = np.array([0,0,0])

        ref_obj_in = np.concatenate([p_obj_in, rot_obj_in])
        ref_obj_f  = np.concatenate([p_obj_f,  rot_obj_f])

        self.traj_time, self.p_obj, self.rpy_obj = generate_trapezoidal_trajectory(
            ref_obj_in, ref_obj_f, self.t0, self.Tf, self.ts,
            v_max=1.0, a_max=1.0
        )
        
        # Profilo QoS per garantire che il messaggio venga ricevuto
        qos_profile = QoSProfile(
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            depth=1
        )
        
        # Publisher per il Path completo (usato dal controllore)
        self.path_pub = self.create_publisher(Path, '/peg_path', qos_profile)

        # Publisher per la singola Pose (usato per la visualizzazione in tempo reale)
        # Sostituito da PoseStamped per includere il frame di riferimento
        self.pose_pub = self.create_publisher(PoseStamped, '/peg_pose', 1)

        qos_ready = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE
        )
        # Subscriber che attende il segnale di pronto dal controllore
        self.ready_subscription = self.create_subscription(
            Bool,
            '/drone_planner_ready',
            self.controller_ready_callback,
            qos_ready)
        
        # Pubblica l'intero path una sola volta per il controllore
        self.publish_full_path()
        
        self.current_index = 0
        self.timer = None
        self.get_logger().info("Peg planner Node avviato. Path completo pubblicato. In attesa del segnale 'ready' dal planner.")


    def publish_full_path(self):
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'world'
        now = self.get_clock().now().to_msg()
        for i in range(len(self.traj_time)):
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = 'world'
            sec = now.sec + int(self.traj_time[i])
            nanosec = now.nanosec + int((self.traj_time[i] - int(self.traj_time[i])) * 1e9)
            if nanosec >= 1_000_000_000:
                sec += 1
                nanosec -= 1_000_000_000
            pose_stamped.header.stamp.sec = sec
            pose_stamped.header.stamp.nanosec = nanosec
            p = self.p_obj[i]
            rpy = self.rpy_obj[i]
            q = RPY_to_quat(rpy[0], rpy[1], rpy[2])
            pose_stamped.pose.position.x = float(p[0])
            pose_stamped.pose.position.y = float(p[1])
            pose_stamped.pose.position.z = float(p[2])
            pose_stamped.pose.orientation.w = float(q[0])
            pose_stamped.pose.orientation.x = float(q[1])
            pose_stamped.pose.orientation.y = float(q[2])
            pose_stamped.pose.orientation.z = float(q[3])
            
            #print(f"Step {i} pos: {p}, rpy: {rpy}")
            #print(f"PoseStamped type: {type(pose_stamped)}")
            path_msg.poses.append(pose_stamped)
        
        #print(f"Path message length: {len(path_msg.poses)}")
        #print(f"First pose: {path_msg.poses[0] if path_msg.poses else 'None'}")
        #print(f"Last pose: {path_msg.poses[-1] if path_msg.poses else 'None'}")
        #print(f"Path message type: {type(path_msg)}")
        self.path_pub.publish(path_msg)
        self.get_logger().info(f"Path pubblicato con {len(path_msg.poses)} pose.")


    def controller_ready_callback(self, msg: Bool):
        """Callback chiamato quando il controllore invia il segnale di pronto."""
        if msg.data and self.timer is None:
            self.get_logger().info("Segnale 'ready' ricevuto dal planner del drone. Avvio pubblicazione pose.")
            self.timer = self.create_timer(self.ts, self.publish_next_pose)
            self.destroy_subscription(self.ready_subscription)


    def publish_next_pose(self):
        """Pubblica la prossima posa dell'oggetto e gestisce la fine della traiettoria."""
        if self.current_index >= len(self.traj_time):
            self.get_logger().info("Fine traiettoria. L'oggetto rimarrà nella posizione finale.")
            self.timer.cancel()
            return

        # Crea un messaggio PoseStamped per includere l'header
        pose_stamped_msg = PoseStamped()
        pose_stamped_msg.header.stamp = self.get_clock().now().to_msg()
        pose_stamped_msg.header.frame_id = 'world'

        p = self.p_obj[self.current_index]
        rpy = self.rpy_obj[self.current_index]
        q = RPY_to_quat(rpy[0], rpy[1], rpy[2])

        pose_stamped_msg.pose.position.x = float(p[0])
        pose_stamped_msg.pose.position.y = float(p[1])
        pose_stamped_msg.pose.position.z = float(p[2])
        pose_stamped_msg.pose.orientation.w = float(q[0])
        pose_stamped_msg.pose.orientation.x = float(q[1])
        pose_stamped_msg.pose.orientation.y = float(q[2])
        pose_stamped_msg.pose.orientation.z = float(q[3])
        
        self.pose_pub.publish(pose_stamped_msg)
        self.current_index += 1


def main(args=None):
    rclpy.init(args=args)
    peg_planner_node = PegPlannerNode()
    rclpy.spin(peg_planner_node)
    peg_planner_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()