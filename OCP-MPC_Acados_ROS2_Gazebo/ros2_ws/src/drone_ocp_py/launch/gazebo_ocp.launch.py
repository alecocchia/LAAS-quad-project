# file: gazebo_ocp.launch.py

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
import os
import xml.etree.ElementTree as ET
from ament_index_python.packages import get_package_share_directory


def get_pose_from_world(world_path, target_keyword):
    """
    Cerca nel file XML del mondo la posa iniziale di un modello specifico.
    Restituisce una tupla (x, y, z) come float.
    """
    try:
        tree = ET.parse(world_path)
        root = tree.getroot()
        
        # Cerca iterativamente in tutti i tag <model> e <include>
        for elem in root.iter():
            if elem.tag in ['model', 'include']:
                name_tag = elem.find('name')
                uri_tag = elem.find('uri')
                
                # Verifica se questo è il modello che stiamo cercando
                is_target = False
                if name_tag is not None and target_keyword in name_tag.text:
                    is_target = True
                elif uri_tag is not None and target_keyword in uri_tag.text:
                    is_target = True
                    
                if is_target:
                    pose_tag = elem.find('pose')
                    if pose_tag is not None and pose_tag.text:
                        # La stringa è del tipo "X Y Z Roll Pitch Yaw"
                        coords = pose_tag.text.strip().split()
                        if len(coords) >= 3:
                            return float(coords[0]), float(coords[1]), float(coords[2]),float(coords[3]), float(coords[4]), float(coords[5]) 
                    
                    # Se trova il modello ma non c'è il tag <pose>, assume l'origine
                    return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                    
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    except Exception as e:
        print(f"[WARNING] Errore nel parsing del world {world_path}: {e}")
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

def generate_launch_description():
    # --- percorsi ---
    pkg_share_dir = get_package_share_directory('mrsim_gazebo_sim')
    sdf_path = os.path.join(pkg_share_dir, 'models', 'mrsim-quad-unico', 'model.sdf')
    world_file = os.path.join(pkg_share_dir, 'worlds', 'example.world')
    bridge_config_file = os.path.join(pkg_share_dir, 'config', 'bridge.yaml')
    rviz_config_file = os.path.join(pkg_share_dir, 'config', 'rviz_config_file.rviz')

    # Estrazione dinamica delle posizioni dal file XML!
    drone_x, drone_y, drone_z, drone_roll, drone_pitch, drone_yaw = get_pose_from_world(world_file, "qr4")
    peg_x, peg_y, peg_z, peg_roll, peg_pitch, peg_yaw  = get_pose_from_world(world_file, "my_peg")

    print(f"[INFO] Lettura dinamica dal World -> Drone: X={drone_x}, Y={drone_y}, Z={drone_z}")
    print(f"[INFO] Lettura dinamica dal World -> Peg: X={peg_x}, Y={peg_y}, Z={peg_z}")

    # --- Parsing dell'XML ---
    tree = ET.parse(sdf_path)
    root = tree.getroot()

    # --- Estrazione dei parametri fisici dall'SDF---
    # Trova tutti i tag <mass> (base + rotori) e li somma
    total_mass = sum([float(m.text) for m in root.findall('.//mass')])
    
    # Trova la prima inerzia e i coefficienti aerodinamici
    ixx = float(root.find('.//ixx').text)
    iyy = float(root.find('.//iyy').text)
    izz = float(root.find('.//izz').text)
    cf = float(root.find('.//cf').text)
    ct = float(root.find('.//ct').text)

    # Trova la posa della camera nel body frame
    camera_pose = root.find('.//sensor[@name="camera"]/pose').text
    cam_x, cam_y, cam_z, cam_roll, cam_pitch, cam_yaw = [float(val) for val in camera_pose.split()]

    #   AGGIUSTARE MODALITA' 3 non funziona
    # --- argomenti ---
    planner_mode_arg = DeclareLaunchArgument(
        'planner_mode', default_value='1',
        description="1=MPC_planner | 2=OCP_planner | 3=test_planner (planner_prova) | 4=offline_planner"
    )

    MPC_controller_arg = DeclareLaunchArgument(
        'MPC_controller', default_value = '1',
        description="1 -> MPC controller utilizzato, 0 -> MPC controller non utilizzato"
    )

    controller_arg = DeclareLaunchArgument(
        'controller', default_value='2',
        description="1=hierarchical (PID_controller) | 2=geometric (geometric_controller)"
    )
    log_file_arg = DeclareLaunchArgument(
        'log_file', default_value='/tmp/pid_run.npz',
        description="File .npz per il replay in offline (planner_mode=2)"
    )
    enable_rviz_arg = DeclareLaunchArgument(
        'enable_rviz', default_value='true',
        description="Apri RViz2"
    )
    # --- argomento opzionale per abilitare l'human node ---
    enable_human_arg = DeclareLaunchArgument(
        'enable_human', default_value='true',
        description="Avvia l'human_goal_node (listener Float64MultiArray → PoseStamped)?"
    )
    enable_human = LaunchConfiguration('enable_human')

    # --- argomento opzionale per abilitare il joystick ---
    enable_joy_arg = DeclareLaunchArgument(
        'enable_joy', default_value='true',
        description="Avvia il nodo joy per leggere il joypad"
    )
    enable_joy = LaunchConfiguration('enable_joy')

    # --- launch configs ---
    planner_mode = LaunchConfiguration('planner_mode')
    controller = LaunchConfiguration('controller')
    MPC_controller = LaunchConfiguration('MPC_controller')
    log_file = LaunchConfiguration('log_file')
    enable_rviz = LaunchConfiguration('enable_rviz')

    # --- helper condizioni ---
    is_planner_mode_1 = IfCondition(PythonExpression([planner_mode, ' == ', '1']))
    is_planner_mode_2 = IfCondition(PythonExpression([planner_mode, ' == ', '2']))
    is_planner_mode_3 = IfCondition(PythonExpression([planner_mode, ' == ', '3']))
    is_planner_mode_4 = IfCondition(PythonExpression([planner_mode, ' == ', '4']))
    is_planner_mode_1_or_2 = IfCondition(PythonExpression(["'", planner_mode, "' in ['1', '2']"]))  # per far partire il peg

    is_ctrl_1 = IfCondition(PythonExpression([
        "'", controller, "'", " == '1' and ",
        "'", MPC_controller, "'", " == '0'"
    ]))
    is_ctrl_2 = IfCondition(PythonExpression([
        "'", controller, "'", " == '2' and ",
        "'", MPC_controller, "'", " == '0'"
    ]))


    omega_ref_world = PythonExpression(["'", planner_mode, "'", " == '3'"]) #planner prova -> omega_ref nel mondo


    # --- ign gazebo ---
    #gz_sim = ExecuteProcess(cmd=['xvfb-run','-a','ign','gazebo','-v','4','-r', world_file])
    gz_sim = ExecuteProcess(cmd=['ign','gazebo','-r','-v','4',world_file])

    # --- bridge ros<->gz ---
    ros_gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='parameter_bridge',
        output='screen', emulate_tty=True,
        arguments=['--ros-args', '-p', f'config_file:={bridge_config_file}'],
        parameters=[{'use_sim_time': True}],
    )

    # --- nodi ROS ---
    # Peg planner (sempre)
    peg_planner = Node(
        package='drone_ocp_py',
        executable='peg_planner_node',
        name='peg_planner_node',
        output='screen', emulate_tty=True,
        parameters=[{
            'use_sim_time': True,
            'peg_start_x': peg_x, 
            'peg_start_y': peg_y, 
            'peg_start_z': peg_z,
            'peg_start_roll': peg_roll, 
            'peg_start_pitch': peg_pitch, 
            'peg_start_yaw': peg_yaw
            }],
        condition=is_planner_mode_1_or_2
    )

    # planner_mode 1: MPC online
    mpc_planner = Node(
        package='drone_ocp_py',
        executable='MPC_planner_node',
        name='MPC_planner_node',
        output='screen', emulate_tty=True,
        parameters=[{
            'use_sim_time': True,
            'control_flag': MPC_controller,
            'mass': total_mass,
            'ixx': ixx,
            'iyy': iyy,
            'izz': izz,
            'cf': cf,
            'ct': ct,
            'cam_x' : cam_x,        # camera in body frame
            'cam_y' : cam_y,
            'cam_z' : cam_z,
            'cam_roll' : cam_roll,
            'cam_pitch' : cam_pitch,
            'cam_yaw' : cam_yaw,
            'start_x': drone_x, 
            'start_y': drone_y, 
            'start_z': drone_z,
            'start_roll': drone_roll, 
            'start_pitch': drone_pitch, 
            'start_yaw': drone_yaw,
        }],
        condition=is_planner_mode_1,
    )

    # Human goal node (sempre pronto, ma utile soprattutto in planner_mode=1)
    human_goal_node = Node(
        package='drone_ocp_py',
        executable='human_goal_node',
        name='human_goal_node',
        output='screen', emulate_tty=True,
        parameters=[{
            'use_sim_time': True,
            'frame_id': 'world',
            'cmd_topic': 'human_goal_vec',   # Float64MultiArray in ingresso
            'goal_topic': 'human_goal',      # PoseStamped in uscita (usato dall’MPC)
        }],
        condition=IfCondition(enable_human),
    )    

    # Nodo Joy (Driver Joystick)
    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        parameters=[{
            'autorepeat_rate': 50.0,
            'deadzone': 0.05,
        }],
        condition=IfCondition(enable_joy),
    )

    # planner_mode 2: OCP online
    ocp_planner = Node(
        package='drone_ocp_py',
        executable='ocp_planner_node',
        name='ocp_planner_node',
        output='screen', emulate_tty=True,
        parameters=[{'use_sim_time': True}],
        condition=is_planner_mode_2,
    )

    # planner_mode 3: test planner (sinusoide)
    planner_prova = Node(
        package='drone_ocp_py',
        executable='planner_prova',
        name='planner_prova',
        output='screen', emulate_tty=True,
        parameters=[{
            'use_sim_time': True,
            # 'amp': 0.5, 'freq_hz': 0.2, 'z0': 1.0, 'rate_hz': 100.0
        }],
        condition=is_planner_mode_3,
    )

    # planner_mode 4: offline loader (replay)   (not working)
    ocp_loader = Node(
        package='drone_ocp_py',
        executable='ocp_offline_loader',
        name='ocp_offline_loader',
        output='screen', emulate_tty=True,
        parameters=[{
            'use_sim_time': True,
            'log_file': log_file,
            'frame_id': 'world',
            'rate_hz': 0.0,          # usa t del log
            'keep_alive_hz': 1.0,    # RViz-friendly
            'publish_wrench': True
        }],
        condition=is_planner_mode_4,
    )

    # Controller 1: PID (hierarchical)
    pid = Node(
        package='drone_ocp_py',
        executable='PID_controller',
        name='PID_controller',
        output='screen', emulate_tty=True,
        parameters=[{
            'use_sim_time': True,
            'omega_ref_world': omega_ref_world
        }],
        condition=is_ctrl_1,
    )

    # Controller 2: geometric
    geom_ctrl = Node(
        package='drone_ocp_py',
        executable='geometric_controller',
        name='geometric_controller',
        output='screen', emulate_tty=True,
        parameters=[{'use_sim_time': True,
                     'omega_ref_world': omega_ref_world,
                    }],
        condition=is_ctrl_2,
    )

    # Logger (sempre)
    logger = Node(
        package='drone_ocp_py',
        executable='logger',
        name='logger',
        output='screen', emulate_tty=True,
        parameters=[{
            'use_sim_time': True,
            'save_path': '/tmp/pid_run.npz',
            'log_hz': 10.0,
            #'save_ref_flag': PythonExpression([planner_mode, ' != ', '2'])
            }],
    )

    # RViz (opzionale)
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen', emulate_tty=True,
        arguments=['-d', rviz_config_file],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(enable_rviz),
    )

    # --- avvio sfalsato (WALL-CLOCK) ---
#    peg_after        = TimerAction(period=1.5, actions=[peg_planner])
    mpc_after        = TimerAction(period=2.0, actions=[mpc_planner])     # se planner_mode==1
#    ocp_after        = TimerAction(period=2.0, actions=[ocp_planner])     # se planner_mode==2
#    test_after       = TimerAction(period=2.0, actions=[planner_prova])   # se planner_mode==3
#    loader_after     = TimerAction(period=2.0, actions=[ocp_loader])      # se planner_mode==4
#    pid_after        = TimerAction(period=0.0, actions=[pid])             # se controller==1
#    geometric_after  = TimerAction(period=2.5, actions=[geom_ctrl])       # se controller==2
    logger_after     = TimerAction(period=0.0, actions=[logger])
#    rviz_after       = TimerAction(period=1.0, actions=[rviz])
#    human_goal_after = TimerAction(period=2.1, actions=[human_goal_node])  # poco dopo mpc_after


    return LaunchDescription([
        planner_mode_arg, controller_arg, log_file_arg, enable_rviz_arg, enable_human_arg, enable_joy_arg, MPC_controller_arg,

        gz_sim,
        ros_gz_bridge,

        peg_planner,

        ocp_planner,
        mpc_after,
        planner_prova,
        ocp_planner,

        pid,
        geom_ctrl,

        human_goal_node,
        joy_node,

        logger,
        rviz,
    ])