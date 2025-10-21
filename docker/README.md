# ros2_docker_scripts

## :package: Overview
This repo contains a template for a ros2 project using Docker. The repo contains a Dockerfile, the scripts to build and run the image and the source code. The result will be a container in which a workspace is built, and the src folder of workspace is binded to the one in the repo. In this way the modifications done inside the container in src folder are reflected into the src folder of the repo, and viceversa...


## Prerequisites
Before setting up the project, you have to install docker. If you already installed docker, go to the next session
```sh
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

# Install the Docker packages
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add docker user
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
```
Then log out and log in.


## :hammer: Build

Clone this repo 
```sh
$ git clone https://github.com/RoboticsLab2024/ros2_docker_scripts.git
```
Open a terminal inside the cloned repo and make all the bash files executable
```sh
$ chmod +x *.sh
```
To build the docker image
```sh
$ ./docker_build_image.sh <IMAGE_NAME>
```
where <IMAGE_NAME> is a name for the image you want to build.

## :white_check_mark: Usage
To run a container
```sh
$ ./docker_run_container.sh <IMAGE_NAME> <CONTAINER_NAME> <FOLDER_NAME>
```
where <IMAGE_NAME> is the name of the image you have just built, while <CONTAINER_NAME> is a name for the container hosting the image. <FOLDER_NAME> is the name of the local folder you want to mount inside the container ROS2 ws.

To connect to the container from another terminal
```sh
$ ./docker_connect.sh 
```
and select the container you want to connect to.

<!-- 
### Notes
1. The container's root password is "user" by default.

2. The container will be automatically destroyed once exited (--rm flag). If you want to attach additional terminals to the container you need to keep it running (docker_run.sh script). You can attach a new terminal by running the following command
```sh
docker exec -it $(docker ps -aqf "name=<CONTAINER_NAME>") bash
```

3. All apt-get performed inside the container will be removed one che container is closed. Please add all new dependacies to the Dockerfile and rebuild the image.

## Run example code
To run the ros2 example code (talker), just execute the following command
```sh
ros2 run cpp_pubsub talker
```

## Development
Once the container is running, you can develop from inside it using VsCode. Open VsCode from your host machine, install the extension Dev Containers, and in command bar select attach to running container. Open the folder /home/user/ros2_ws/src, which is the binded folder. 
   
-->
   
   
   
   
   
   
   
   
   
   
   
   
   

