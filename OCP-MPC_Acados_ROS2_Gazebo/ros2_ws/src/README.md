# src
src folder in ros2_ws

# Building and sourcing the environment
cd ~/ros2_ws
colcon build
source install/setup.bash

# Launch
  ros2 launch drone_ocp_py gazebo_ocp.launch.py

# Launchfile arguments:
  - planner_mode (OCP (static optimization problem), MPC, offline (not working), test (simusoidal test))
  - controller (hierarchical PID (not working) or geometric)
  - MPC_controller : MPC acts both as planner and as controller. If MPC_controller is 1, controller arg will be ignored.
  - log_file: file on which simulation results are saved
  - enable_rviz (default true)
  - enable_human (default true)
  
## NODES

  # Human goal
    If MPC is used (at least as planner), it is possible to impose an external reference on a new terminal.
    Steps:
      - open a new terminal
     - go in docker root directory
     - ./docker_connect.sh 
     - select container that is currently running
     - impose reference with:
         ros2 topic pub -1 /human_goal_vec std_msgs/msg/Float64MultiArray "{data: [radius, dist[m], azimuth[rad], polar angle[rad]}"
    This reference is meant to be the one between drone and object with respect to the drone PoV.

  # Logger
    This node acts as logger -> it listens to and saves simulation results over log_file specified in launchfile

## Results 
  # plot_script.py
    Just run it on python. It shows results saved in log_file. More quantities should be saved and shown (acc, jerk, etc.)
  # ros_bag_plotter.py
    It is thought for bag_files plotting, but rosbag has not be used.

## Utility files
  common.py, drone_model.py, drone_MPC_settings.py, drone_OCP_settings.py, drone_simulation.py
    These are included by main scripts (es. MPC_planner_node.py or ocp_planner_node.py)
         
     
