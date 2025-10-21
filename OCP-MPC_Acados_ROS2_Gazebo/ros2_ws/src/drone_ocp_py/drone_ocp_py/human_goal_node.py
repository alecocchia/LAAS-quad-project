#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped

def euler_to_quat(roll: float, pitch: float, yaw: float):
    """ZYX convention (roll=X, pitch=Y, yaw=Z)."""
    cy, sy = math.cos(yaw * 0.5),   math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5),  math.sin(roll * 0.5)
    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return w, x, y, z

class HumanGoalNode(Node):
    """
    Converte un Float64MultiArray in PoseStamped e lo pubblica una sola volta su 'human_goal'.
    Formati accettati:
      - [r, pan, tilt, yaw]                (roll=pitch=0)
      - [r, pan, tilt, roll, pitch, yaw]

    Usage:
    ros2 topic pub -1 /human_goal_vec std_msgs/msg/Float64MultiArray "{data: [2.8, 0.2, 0.0, 1.57]}"
    """
    def __init__(self):
        super().__init__('human_goal_node')

        # Parametri
        self.declare_parameter('frame_id', 'world')
        self.declare_parameter('cmd_topic', 'human_goal_vec')   # Float64MultiArray in ingresso
        self.declare_parameter('goal_topic', 'human_goal')      # PoseStamped in uscita

        frame_id  = str(self.get_parameter('frame_id').value) or 'world'
        cmd_topic = str(self.get_parameter('cmd_topic').value) or 'human_goal_vec'
        out_topic = str(self.get_parameter('goal_topic').value) or 'human_goal'

        # QoS: affidabile, buffer minimo
        qos_in = QoSProfile(depth=1)
        qos_in.reliability = QoSReliabilityPolicy.RELIABLE
        qos_in.history = QoSHistoryPolicy.KEEP_LAST

        qos_out = QoSProfile(depth=1)
        qos_out.reliability = QoSReliabilityPolicy.RELIABLE
        qos_out.history = QoSHistoryPolicy.KEEP_LAST

        # Publisher e subscriber
        self.goal_pub = self.create_publisher(PoseStamped, out_topic, qos_out)
        self.sub = self.create_subscription(
            Float64MultiArray, cmd_topic, self.cmd_cb, qos_in
        )

        self._frame_id = frame_id
        self.get_logger().info(
            f"human_goal_node attivo. Ascolto '{cmd_topic}' (Float64MultiArray) "
            f"→ pubblico '{out_topic}' (PoseStamped). frame_id='{frame_id}'."
        )

    def cmd_cb(self, msg: Float64MultiArray):
        data = list(msg.data) if msg.data is not None else []
        n = len(data)

        if n not in (4, 6):
            self.get_logger().warn(
                f"Comando ignorato: attesi 4 o 6 elementi, ricevuti {n}. "
                "Formati: [r,pan,tilt,yaw] oppure [r,pan,tilt,roll,pitch,yaw]."
            )
            return

        # Parsifica
        r, pan, tilt = float(data[0]), float(data[1]), float(data[2])
        if n == 4:
            roll, pitch, yaw = 0.0, 0.0, float(data[3])
        else:
            roll, pitch, yaw = float(data[3]), float(data[4]), float(data[5])

        # Costruisci PoseStamped (interpretando pos = [r,pan,tilt])
        out = PoseStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self._frame_id
        out.pose.position.x = r
        out.pose.position.y = pan
        out.pose.position.z = tilt
        qw, qx, qy, qz = euler_to_quat(roll, pitch, yaw)
        out.pose.orientation.w = qw
        out.pose.orientation.x = qx
        out.pose.orientation.y = qy
        out.pose.orientation.z = qz

        # Pubblica UNA VOLTA (l'MPC gestisce l'hold)
        self.goal_pub.publish(out)
        self.get_logger().info(
            f"human_goal pubblicato: r={r:.2f}, pan={pan:.2f}, tilt={tilt:.2f}, "
            f"rpy=({roll:.2f},{pitch:.2f},{yaw:.2f})"
        )

def main():
    rclpy.init()
    node = HumanGoalNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
