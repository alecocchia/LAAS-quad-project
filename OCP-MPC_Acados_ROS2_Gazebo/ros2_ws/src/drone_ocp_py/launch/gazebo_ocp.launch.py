# file: gazebo_ocp.launch.py

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

################# CAMBIARE : DARE LA POSSIBILITA' TRAMITE PARAMETRO DI DECIDERE SE MPC PUBBLICA SU OPTIMAL_WRENCH O WRENCH_CMD
################# COS'ALTRO AGGIUSTARE: PROBABILE DISCONTINUITÀ A +-PI NEL PLANNER (provare a non passare per rpy -> Rot mat direttamente e q)
################# INSERIRE CAMERA
################# DECIDERE COERENTEMENTE LE FREQUENZE E TEMPI DI PREDIZIONE ETC.
################# INTRODURRE COMANDI CONTROLLER
def generate_launch_description():
    # --- percorsi ---
    pkg_share_dir = get_package_share_directory('mrsim_gazebo_sim')
    world_file = os.path.join(pkg_share_dir, 'worlds', 'example.world')
    bridge_config_file = os.path.join(pkg_share_dir, 'config', 'bridge.yaml')
    rviz_config_file = os.path.join(pkg_share_dir, 'config', 'rviz_config_file.rviz')

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

    is_ctrl_1 = IfCondition(PythonExpression([
        "'", controller, "'", " == '1' and ",
        "'", MPC_controller, "'", " == 'false'"
    ]))
    is_ctrl_2 = IfCondition(PythonExpression([
        "'", controller, "'", " == '2' and ",
        "'", MPC_controller, "'", " == 'false'"
    ]))


    omega_ref_world = PythonExpression(["'", planner_mode, "'", " == '3'"]) #planner prova -> omega_ref nel mondo


    # --- ign gazebo ---
    #gz_sim = ExecuteProcess(cmd=['xvfb-run','-a','ign','gazebo','-v','4','-r', world_file])
    gz_sim = ExecuteProcess(cmd=['ign','gazebo','-v','4','-r',world_file])

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
        parameters=[{'use_sim_time': True}],
    )

    # planner_mode 1: MPC online
    mpc_planner = Node(
        package='drone_ocp_py',
        executable='MPC_planner_node',
        name='MPC_planner_node',
        output='screen', emulate_tty=True,
        parameters=[{
            'use_sim_time': True,
            'control_flag': MPC_controller
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
            'log_hz': 100.0,
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
    peg_after        = TimerAction(period=1.5, actions=[peg_planner])
    mpc_after        = TimerAction(period=2.0, actions=[mpc_planner])     # se planner_mode==1
    ocp_after        = TimerAction(period=2.0, actions=[ocp_planner])     # se planner_mode==2
    test_after       = TimerAction(period=2.0, actions=[planner_prova])   # se planner_mode==3
    loader_after     = TimerAction(period=2.0, actions=[ocp_loader])      # se planner_mode==4
    pid_after        = TimerAction(period=0.0, actions=[pid])             # se controller==1
    geometric_after  = TimerAction(period=2.5, actions=[geom_ctrl])       # se controller==2
    logger_after     = TimerAction(period=0.0, actions=[logger])
    rviz_after       = TimerAction(period=1.0, actions=[rviz])
    human_goal_after = TimerAction(period=2.1, actions=[human_goal_node])  # poco dopo mpc_after

    return LaunchDescription([
        planner_mode_arg, controller_arg, log_file_arg, enable_rviz_arg, enable_human_arg, MPC_controller_arg,

        gz_sim,
        ros_gz_bridge,

        peg_after,

        ocp_after,
        mpc_after,
        test_after,
        loader_after,

        pid_after,
        geometric_after,

        human_goal_after,

        logger_after,
        rviz_after,
    ])
