FROM osrf/ros:humble-desktop

# Uncomment if needed for apt-get date errors
# RUN echo "Acquire::Check-Valid-Until \"false\";\nAcquire::Check-Date \"false\";" | cat > /etc/apt/apt.conf.d/10no--check-valid-until

# Update apt-get and install some basic utilities needed later
RUN apt-get update && apt-get install -y \
    lsb-release gnupg curl software-properties-common apt-transport-https \
    git \
    build-essential \
    cmake \
    pkg-config \
    libusb-dev \
    iputils-ping \
    net-tools \
    iproute2 \
    vim \
    nano \
    gedit \
    htop \
    screen \
    tmux \
    wget \
    unzip \
    gdb \
    valgrind \
    libasio-dev \
    libtinyxml2-dev \
    libconsole-bridge-dev \
    libgtest-dev \
    libceres-dev \
    python3-rosdep \
    python3-vcstool \
    python3-colcon-common-extensions \
    terminator \
    tree

# Inizio installazione di Visual Studio Code tramite repository APT
RUN wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg && \
    install -D -o root -g root -m 644 packages.microsoft.gpg /etc/apt/keyrings/packages.microsoft.gpg && \
    rm packages.microsoft.gpg

RUN sh -c 'echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list'

RUN apt-get update && \
    apt-get install -y code

# Add OSRF GPG key for Gazebo
RUN wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/osrf-gazebo.gpg

# Add OSRF Gazebo stable repository for Jammy (Ubuntu 22.04)
# (Aggiunta esplicita dell'architettura per prevenire warnings)
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/osrf-gazebo.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/osrf-gazebo.list > /dev/null

# -- Inizio blocco installazione dipendenze Ignition esatte --
# Questo passaggio è CRUCIALE per garantire che le versioni specifiche siano installate.
# Usiamo apt-get update prima di ogni blocco per assicurarci che la lista dei pacchetti sia fresca.
RUN apt-get update && apt-get install -y \
    ignition-fortress \
    ignition-tools
    #\
    #libignition-cmake2-dev \
    #libignition-common4-dev \
    #libignition-fuel-tools7-dev \
    #libignition-gazebo6-dev \
    #libignition-launch5-dev \
    #libignition-math6-dev \
    #libignition-msgs8-dev \
    #libignition-physics5-dev \
    #libignition-plugin-dev \
    #libignition-rendering6-dev \
    #libignition-sensors6-dev \
    #libignition-transport11-dev \
    #libignition-utils1-dev \
    #libsdformat12-dev



# -- Fine blocco installazione dipendenze Ignition esatte --

# Install python3-pip (pip3)
RUN apt-get update && apt-get install -y python3-pip

# Install casadi via pip (user install disabled, as siamo root)
RUN pip3 install casadi

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=:0
ENV HOME=/home/user
ENV ROS_DISTRO=humble

# Add non-root user with UID and GID from build args
ARG USER_ID
ARG GROUP_ID
RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
RUN echo "user:user" | chpasswd
RUN echo "user ALL=(ALL:ALL) ALL" >> /etc/sudoers
USER user

# ROS2 workspace creation and compilation
RUN mkdir -p ${HOME}/ros2_ws/src
WORKDIR ${HOME}/ros2_ws
SHELL ["/bin/bash", "-c"]

# Add script source to .bashrc (Manteniamo solo le fonti essenziali)
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash;" >>  ${HOME}/.bashrc
RUN echo "source ${HOME}/ros2_ws/install/local_setup.bash;" >>  ${HOME}/.bashrc
RUN echo "source /usr/share/colcon_cd/function/colcon_cd.sh" >> ${HOME}/.bashrc
RUN echo "source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash" >> ${HOME}/.bashrc

USER root

# ROS packages (come nel tuo Dockerfile originale)
RUN apt-get update && apt-get install -y \
    ros-humble-ros-ign-bridge \
    ros-humble-ros-gz \
    ros-humble-controller-manager \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-ign-ros2-control \
    ros-humble-ign-ros2-control-demos \
    ros-humble-joint-state-publisher-gui \
    ros-humble-xacro \
    ros-humble-usb-cam \
    ros-humble-image-pipeline \
    ros-humble-tf-transformations \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-robot-localization


# === ACADOS Installation ===
USER root

# Set working directory
WORKDIR /opt

# Clone acados with submodules
RUN git clone --recursive https://github.com/acados/acados.git

# Build and install
WORKDIR /opt/acados
RUN mkdir -p build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/opt/acados -DACADOS_WITH_QPOASES=ON .. && \
    make -j$(nproc) && make install

# Set environment variables
ENV ACADOS_SOURCE_DIR="/opt/acados"
ENV ACADOS_INSTALL_DIR="/opt/acados"

# INIZIO MODIFICHE
# Abilita OpenMP (minimo indispensabile)
RUN apt-get update && apt-get install -y libomp-dev

# Riconfigura e reinstalla acados con OpenMP ON, mantenendo la stessa install dir
RUN cd /opt/acados && \
    mkdir -p build && cd build && \
    cmake .. \
      -DACADOS_WITH_OPENMP=ON \
      -DBUILD_SHARED_LIBS=ON \
      -DACADOS_INSTALL_DIR=/opt/acados \
      -DBLASFEO_TARGET=GENERIC \
      -DHPIPM_TARGET=GENERIC && \
    make -j"$(nproc)" && make install

# (opzionale) Limita/assegna i thread OMP
ENV OMP_NUM_THREADS=4

# FINE MODIFICHE
# Optional: Install Python interface (acados_template)
RUN pip3 install /opt/acados/interfaces/acados_template

# Scarica t_renderer, rinominalo, rendilo eseguibile e spostalo in /opt/acados/bin
RUN curl -L https://github.com/acados/tera_renderer/releases/download/v0.2.0/t_renderer-v0.2.0-linux-amd64 \
    -o /opt/acados/bin/t_renderer && \
    chmod +x /opt/acados/bin/t_renderer

RUN apt-get update 
WORKDIR ${HOME}/ros2_ws

# Pulizia finale dei cache apt per ridurre la dimensione dell'immagine
RUN rm -rf /var/lib/apt/lists/*
USER user
