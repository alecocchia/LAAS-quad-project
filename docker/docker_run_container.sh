#!/bin/bash

# Gestione degli argomenti di aiuto e mancanza di argomenti
if [[ "$@" == "--help" || "$@" == "-h" ]]; then
    echo "Usage: $0 [IMAGE_NAME] [CONTAINER_NAME (optional)]"
    echo "  [IMAGE_NAME]     : Nome dell'immagine Docker da avviare (es. ros2-gz)"
    echo "  [CONTAINER_NAME] : (Opzionale) Nome da assegnare al container. Default = Gaetano"
    exit 0
elif [[ "$#" -eq 0 ]]; then
    echo "Usage: $0 [IMAGE_NAME] [CONTAINER_NAME (optional)]"
    exit 1
fi

# Assegnazione degli argomenti
IMAGE_NAME="$1"
CONTAINER_NAME="${2:-Gaetano}"   # Se non viene passato, usa "Gaetano"

HOST_ROS2_WS_FOLDER="ros2_ws"
HOST_ROS2_SRC_FOLDER="${HOST_ROS2_WS_FOLDER}/src"

# Percorso completo della cartella di sviluppo ROS 2 sull'host
HOST_ROS2_WS_PATH="/home/$USER/$HOST_ROS2_WS_FOLDER"
HOST_ROS2_SRC_PATH="$HOST_ROS2_WS_PATH/src"

# Percorso della workspace ROS 2 all'interno del container
CONTAINER_ROS2_WS_PATH="/home/user/${HOST_ROS2_WS_FOLDER}"
CONTAINER_ROS2_SRC_PATH="$CONTAINER_ROS2_WS_PATH/src"

# Percorsi per i file bag
HOST_BAG_FILES_PATH="/home/$USER/bag_files"
CONTAINER_BAG_FILES_PATH="/home/user/bag_files"

# Controlla se la cartella di sviluppo sull'host esiste, altrimenti creala
if [ ! -d "$HOST_ROS2_WS_PATH" ]; then
    echo "[WARNING] Cartella di sviluppo '$HOST_ROS2_WS_PATH' non trovata, creandone una nuova."
    mkdir -p "$HOST_ROS2_WS_PATH"
fi

if [ ! -d "$HOST_BAG_FILES_PATH" ]; then
    echo "[INFO] Cartella per i file bag '$HOST_BAG_FILES_PATH' non trovata, creandone una nuova."
    mkdir -p "$HOST_BAG_FILES_PATH"
fi

# Configurazione per X11 forwarding
xhost +local:root

# --- Determinazione degli argomenti GPU ---
DOCKER_GPU_ARGS=""
if docker info | grep -q "Runtimes: .*nvidia"; then
    echo "Rilevata GPU NVIDIA. Avvio del container con supporto NVIDIA (--gpus all)."
    DOCKER_GPU_ARGS="--gpus all"
else
    echo "Nessuna GPU NVIDIA rilevata. Avvio con accesso generico a /dev/dri."
fi

# --- Impostazioni per il container ---
CONTAINER_ROS2_WS_HOME="/home/user/ros2_ws"
CONTAINER_ROS2_WS_SRC="${CONTAINER_ROS2_WS_HOME}/src"
CONTAINER_ROS2_WS_INSTALL="${CONTAINER_ROS2_WS_HOME}/install"
CONTAINER_ROS2_SIM="${CONTAINER_ROS2_WS_INSTALL}/mrsim_gazebo_sim/share/mrsim_gazebo_sim"
CONTAINER_ROS2_MODELS="${CONTAINER_ROS2_SIM}/models"
CONTAINER_ROS2_WORLDS="${CONTAINER_ROS2_SIM}/worlds"

# Paths Gazebo
GZ_RESOURCE_PATH="${CONTAINER_ROS2_MODELS}:${CONTAINER_ROS2_WORLDS}:${CONTAINER_ROS2_WS_SRC}"
GZ_PLUGIN_PATH="${CONTAINER_ROS2_WS_INSTALL}/mrsim_gazebo_sim/lib/mrsim_gazebo_sim"
LIBRARY_PATH="${GZ_PLUGIN_PATH}"

# --- Avvio del container Docker ---
echo "Avvio del container Docker '${CONTAINER_NAME}' dall'immagine '${IMAGE_NAME}'..."

docker stop "${CONTAINER_NAME}" &> /dev/null
docker rm "${CONTAINER_NAME}" &> /dev/null

docker run -it --rm \
    --name="${CONTAINER_NAME}" \
    --net=host \
    -e DISPLAY="${DISPLAY}" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v /dev/dri:/dev/dri \
    -v "${HOST_ROS2_SRC_PATH}:${CONTAINER_ROS2_SRC_PATH}" \
    -v "${HOST_BAG_FILES_PATH}:${CONTAINER_BAG_FILES_PATH}" \
    --workdir="${CONTAINER_ROS2_WS_PATH}" \
    -e QT_X11_NO_MITSHM=1 \
    -e IGN_GAZEBO_RESOURCE_PATH="${GZ_RESOURCE_PATH}" \
    -e IGN_GAZEBO_SYSTEM_PLUGIN_PATH="${GZ_PLUGIN_PATH}:/usr/lib/x86_64-linux-gnu/ign-gazebo-6/plugins" \
    -e _COLCON_CD_ROOT="${CONTAINER_ROS2_WS_HOME}" \
    -e LD_LIBRARY_PATH="/opt/acados/lib:${LIBRARY_PATH}:/usr/lib/x86_64-linux-gnu/ign-gazebo-6/plugins:\$LD_LIBRARY_PATH" \
    ${DOCKER_GPU_ARGS} \
    "${IMAGE_NAME}" \
    bash -l

xhost -local:root
echo "Container terminato."
