# drone_sim_launch.py MODIFICATO

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    return LaunchDescription([
        # --- Argomenti ---
        DeclareLaunchArgument(
            'world_file_name',
            default_value='example.world',
            description='World file to load'
        ),

        # --- Avvio Gazebo ---
        # Gazebo caricherà il mondo E i modelli definiti al suo interno
        ExecuteProcess(
            cmd=['ign', 'gazebo', LaunchConfiguration('world_file_name'), '-r', '--verbose'],
            output='screen',
        ),

        # --- IL NODO PER LO SPAWN È STATO RIMOSSO ---
    ])