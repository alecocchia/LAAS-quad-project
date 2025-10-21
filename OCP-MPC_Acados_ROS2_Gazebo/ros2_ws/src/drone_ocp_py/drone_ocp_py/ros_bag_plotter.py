import rosbag2_py
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped
from rclpy.serialization import deserialize_message
import matplotlib.pyplot as plt

BAG_PATH = "/home/user/bag_files/my_comparison_bag"
OPTIMAL_PATH_TOPIC = "/optimal_drone_path"
DRONE_POSES_TOPIC = "/drone_pose"
ODOMETRY_TOPIC = "/odometry"

def open_reader():
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=BAG_PATH, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions('', '')
    reader.open(storage_options, converter_options)
    return reader

# --- Prima scansione per trovare t0 ---
reader = open_reader()

first_optimal_time = None
first_drone_time = None
first_odom_time = None

while reader.has_next():
    topic, data, _ = reader.read_next()

    if topic == OPTIMAL_PATH_TOPIC and first_optimal_time is None:
        path_msg = deserialize_message(data, Path)
        first_optimal_time = path_msg.header.stamp.sec + path_msg.header.stamp.nanosec * 1e-9

    if topic == DRONE_POSES_TOPIC and first_drone_time is None:
        pose_msg = deserialize_message(data, PoseStamped)
        first_drone_time = pose_msg.header.stamp.sec + pose_msg.header.stamp.nanosec * 1e-9

    if topic == ODOMETRY_TOPIC and first_odom_time is None:
        odom_msg = deserialize_message(data, Odometry)
        first_odom_time = odom_msg.header.stamp.sec + odom_msg.header.stamp.nanosec * 1e-9

    if first_optimal_time and first_drone_time and first_odom_time:
        break

t0_candidates = [t for t in [first_optimal_time, first_drone_time, first_odom_time] if t is not None]
t0 = min(t0_candidates) if t0_candidates else 0.0
# --- Seconda scansione: estrazione dati ---
reader = open_reader()

dt = 0.01  # intervallo fisso per path

t_opt, x_opt_vals, y_opt_vals, z_opt_vals = [], [], [], []
t_drone, x_vals, y_vals, z_vals = [], [], [], []
t_meas, x_meas, y_meas, z_meas = [], [], [], []

got_optimal_path = False

while reader.has_next():
    topic, data, _ = reader.read_next()

    if topic == OPTIMAL_PATH_TOPIC and not got_optimal_path:
        path_msg = deserialize_message(data, Path)
        rel_time = first_optimal_time
        # Generiamo tempo relativo basato su dt
        for pose_stamped in path_msg.poses:
                t_opt.append(rel_time)
                x_opt_vals.append(pose_stamped.pose.position.x)
                y_opt_vals.append(pose_stamped.pose.position.y)
                z_opt_vals.append(pose_stamped.pose.position.z)
                rel_time += dt
        got_optimal_path = True

    if topic == DRONE_POSES_TOPIC:
        pose_msg = deserialize_message(data, PoseStamped)
        t_stamp = pose_msg.header.stamp.sec + pose_msg.header.stamp.nanosec * 1e-9
        rel_time = t_stamp
        t_drone.append(rel_time)
        x_vals.append(pose_msg.pose.position.x)
        y_vals.append(pose_msg.pose.position.y)
        z_vals.append(pose_msg.pose.position.z)

    if topic == ODOMETRY_TOPIC:
        odom_msg = deserialize_message(data, Odometry)
        t_stamp = odom_msg.header.stamp.sec + odom_msg.header.stamp.nanosec * 1e-9
        rel_time = t_stamp
        if rel_time >= first_odom_time :
            t_meas.append(rel_time)
            x_meas.append(odom_msg.pose.pose.position.x)
            y_meas.append(odom_msg.pose.pose.position.y)
            z_meas.append(odom_msg.pose.pose.position.z)


# --- Plot dei risultati ---

plt.figure()
plt.plot(t_opt, x_opt_vals, label='x_opt')
plt.plot(t_drone, x_vals, label='x_sim')
plt.plot(t_meas, x_meas, label='x_meas')
plt.xlabel('Tempo [s]')
plt.ylabel('Posizione X [m]')
plt.title('X: Traiettoria ottima vs simulazione vs odometria')
plt.grid(True)
plt.legend()

plt.figure()
plt.plot(t_opt, y_opt_vals, label='y_opt')
plt.plot(t_drone, y_vals, label='y_sim')
plt.plot(t_meas, y_meas, label='y_meas')
plt.xlabel('Tempo [s]')
plt.ylabel('Posizione Y [m]')
plt.title('Y: Traiettoria ottima vs simulazione vs odometria')
plt.grid(True)
plt.legend()

plt.figure()
plt.plot(t_opt, z_opt_vals, label='z_opt')
plt.plot(t_drone, z_vals, label='z_sim')
plt.plot(t_meas, z_meas, label='z_meas')
plt.xlabel('Tempo [s]')
plt.ylabel('Posizione Z [m]')
plt.title('Z: Traiettoria ottima vs simulazione vs odometria')
plt.grid(True)
plt.legend()

plt.show()
