import rclpy
from rclpy.node import Node
import numpy as np

from drone_ocp_py.planner import generate_trapezoidal_trajectory


class MainNode(Node):
    def __init__(self):
        super().__init__('main_node')
        self.get_logger().info("Nodo drone_OCP attivato")

        # Stato iniziale e finale: p + rpy
        x0 = np.array([0.0, 0.0, 0.0,     # pos iniziale
                       0.0, 0.0, 0.0])    # rpy iniziale
        x_ref = np.array([2.0, 2.0, 1.0,  # pos finale
                          0.0, 0.0, 1.57])  # rpy finale (yaw 90°)

        # Parametri tempo
        t0 = 0.0
        tf = 5.0
        dt = 0.1

        # Genera traiettoria
        traj = generate_trapezoidal_trajectory(x0, x_ref, t0, tf, dt)

        self.get_logger().info(f"Traiettoria generata con {len(traj[0])} punti.")


def main(args=None):
    rclpy.init(args=args)
    node = MainNode()
    rclpy.spin_once(node, timeout_sec=0.5)  # Evita ciclo infinito
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
