- folder for docker configuration (dockerfile, docker run and docker connect)
- Folder for OCP/MPC with AcadosSimulator
- Folder for OCP/MPC with Gazebo Simulator
Notes:
- Hierarchical PID controller still not working correctly
- With OCP planner a reference change from pi/2 to pi still sometimes is taken as longest distance (pi/2 -> -pi)


Usage of docker:
1) docker_build script for building the docker Image. Inside there is a full ROS2-Ignition Gazebo v6 - ACADOS framework.
    Usage: execute specifying the image name
2) docker_run script for running a container (each container is destroyed when exited). 
    Usage: execute specifying image name. There's no need to specify the container's name; it will be chosen automatically as a fixed name (useful for VSCode)
3) docker_connect script for connecting to a running container. Just execute the bash file and it will be asked the name of container to connect.
