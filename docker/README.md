# Usage of docker:
# 1) docker_build script for building the docker Image. Inside there is a full ROS2-Ignition Gazebo v6 - ACADOS framework.
    Usage: execute specifying the image name
# 2) docker_run script for running a container (each container is destroyed when exited). 
    Usage: execute specifying image name. There's no need to specify the container's name; it will be chosen automatically as a fixed name (useful for VSCode if you want to attach it to the container)
# 3) docker_connect script for connecting to a running container. Just execute the bash file and it will be asked the name of container to connect.
