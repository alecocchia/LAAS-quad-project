import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Wrench
import numpy as np

class OpenLoopTest(Node):
    def __init__(self):
        super().__init__('openLoop_test')

        # --- Parametri base ---
        self.m = 1.28
        self.g = 9.81
        self.pub_rate_hz =200.0
        self.cmd_topic  = '/wrench_cmd'

        # Modulazione opzionale (metti ampiezza_forza=0 per disattivare)
        self.num_camp = 1000
        alpha = np.linspace(0, 2*np.pi, self.num_camp)
        ampiezza_forza = 0.0  # N
        self.mod_signal = ampiezza_forza * np.sin(alpha)
        self.count = 0

        # Publisher Wrench (interpretabile dal tuo plugin in BODY frame)
        self.wrench_pub = self.create_publisher(Wrench, self.cmd_topic, 10)

        # Timer
        self.timer = self.create_timer(1.0 / self.pub_rate_hz, self.publish_test_wrench)

        self.get_logger().info(f"ControllerTestPublisher: m={self.m} kg, g={self.g} m/s^2")

    def publish_test_wrench(self):
        # Hover richiesto in world
        Fz_world_des = self.m * self.g

        # Supp. drone orizzontale
        Fz_body = Fz_world_des

        # Modulazione opzionale (default 0)
        if self.count >= self.num_camp:
            self.count = 0
        Fz_body += float(self.mod_signal[self.count])
        self.count += 1

        # Costruisci il Wrench in BODY frame: solo thrust e nessuna coppia
        w = Wrench()
        w.force.x = 0.0
        w.force.y = 0.0
        w.force.z = float(Fz_body)
        w.torque.x = 0.0
        w.torque.y = 0.0
        w.torque.z = 0.0

        self.wrench_pub.publish(w)
        # log throttle (una volta al secondo)
        #self.get_logger().info(1.0, f'Fz_body={Fz_body:.2f} N (R33={self.R33:.3f})')

def main(args=None):
    rclpy.init(args=args)
    node = OpenLoopTest()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()