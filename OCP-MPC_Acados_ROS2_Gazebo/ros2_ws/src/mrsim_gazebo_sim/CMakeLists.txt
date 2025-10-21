cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(mrsim_gazebo_sim)

#include_directories("/usr/include")
#link_directories("/usr/lib/x86_64-linux-gnu")

# Required for ROS 2 packages
find_package(ament_cmake REQUIRED)

# ROS 2 Core
find_package(rclcpp REQUIRED)

# ROS 2 Message types for publishing/subscribing
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# Ignition Gazebo 6 (Gazebo Fortress) dependencies
find_package(ignition-gazebo6 REQUIRED)
find_package(ignition-math6 REQUIRED)
find_package(ignition-msgs8 REQUIRED) # Contiene gz/msgs/wrench.hh
find_package(ignition-plugin1 REQUIRED COMPONENTS register)
find_package(ignition-common4 REQUIRED)
find_package(ignition-transport11 REQUIRED)
#find_package(tinyxml2 REQUIRED)

# Sezione per SDF
find_package(PkgConfig REQUIRED)
pkg_check_modules(SDFORMAT REQUIRED sdformat12)

# Per la parte ros_gz_sim (il bridge ROS 2 <-> Ignition)
find_package(ros_gz_sim REQUIRED)

# Add the plugin library
add_library(mrsim_drone_ign_plugin SHARED
  src/mrsim_drone_ign_plugin.cpp
)

# Add the plugin library
add_library(mrsim_peg_pose_ign_plugin SHARED
  src/mrsim_peg_pose_ign_plugin.cpp
)


# Set standard C++ flags for the plugin
target_compile_features(mrsim_drone_ign_plugin PUBLIC cxx_std_17)
target_compile_options(mrsim_drone_ign_plugin PRIVATE
  -Wall
  -Wextra
  -Wpedantic
)

# Set standard C++ flags for the plugin
target_compile_features(mrsim_peg_pose_ign_plugin PUBLIC cxx_std_17)
target_compile_options(mrsim_peg_pose_ign_plugin PRIVATE
  -Wall
  -Wextra
  -Wpedantic
)

## Directory di Inclusione

target_include_directories(mrsim_drone_ign_plugin  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> # Optional, if you have your own headers
  $<INSTALL_INTERFACE:include>
  # Directory di inclusione per ROS 2 core e messaggi
  ${rclcpp_INCLUDE_DIRS}
  ${std_msgs_INCLUDE_DIRS}
  ${geometry_msgs_INCLUDE_DIRS}
  ${nav_msgs_INCLUDE_DIRS}
  ${sensor_msgs_INCLUDE_DIRS}
  ${tf2_ros_INCLUDE_DIRS}
  ${tf2_geometry_msgs_INCLUDE_DIRS}
  # Directory di inclusione per sdformat12
  ${SDFORMAT_INCLUDE_DIRS}
  # IMPORTANT: Aggiungi qui il percorso di inclusione principale di Ignition Gazebo.
  # Questo rende 'gz/sim/components/ChildLinkName.hh' trovabile.
  ${ignition-gazebo6_INCLUDE_DIRS}
  # AGGIUNTO: Directory di inclusione per Ignition Msgs per tipi come gz/msgs/wrench.hh
  ${ignition-msgs8_INCLUDE_DIRS}
)

target_include_directories(mrsim_peg_pose_ign_plugin  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> # Optional, if you have your own headers
  $<INSTALL_INTERFACE:include>
  # Directory di inclusione per ROS 2 core e messaggi
  ${rclcpp_INCLUDE_DIRS}
  ${std_msgs_INCLUDE_DIRS}
  ${geometry_msgs_INCLUDE_DIRS}
  ${nav_msgs_INCLUDE_DIRS}
  ${sensor_msgs_INCLUDE_DIRS}
  ${tf2_ros_INCLUDE_DIRS}
  ${tf2_geometry_msgs_INCLUDE_DIRS}
  # Directory di inclusione per sdformat12
  ${SDFORMAT_INCLUDE_DIRS}
  # IMPORTANT: Aggiungi qui il percorso di inclusione principale di Ignition Gazebo.
  # Questo rende 'gz/sim/components/ChildLinkName.hh' trovabile.
  ${ignition-gazebo6_INCLUDE_DIRS}
  # AGGIUNTO: Directory di inclusione per Ignition Msgs per tipi come gz/msgs/wrench.hh
  ${ignition-msgs8_INCLUDE_DIRS}
)

## Librerie di Linkage

ament_target_dependencies(mrsim_drone_ign_plugin
  rclcpp
  # RIMOSSO: ignition-gazebo6::gazebo - Non esiste come target nel tuo ambiente
  ignition-math6
  ignition-msgs8 # Già correttamente collegato qui
  ignition-plugin1
  ignition-common4
  ignition-transport11
  #${SDFORMAT_LIBRARIES}
  #${ros_gz_sim_LIBRARIES}
  std_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
  tf2_ros
  tf2_geometry_msgs
)

ament_target_dependencies(mrsim_peg_pose_ign_plugin
  rclcpp
  # RIMOSSO: ignition-gazebo6::gazebo - Non esiste come target nel tuo ambiente
  ignition-math6
  ignition-msgs8 # Già correttamente collegato qui
  ignition-plugin1
  ignition-common4
  ignition-transport11
  #${SDFORMAT_LIBRARIES}
  #${ros_gz_sim_LIBRARIES}
  std_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
  tf2_ros
  tf2_geometry_msgs
)

# Install the plugin so Gazebo can find it
install(TARGETS mrsim_drone_ign_plugin
  DESTINATION lib/${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}
)

# Install the plugin so Gazebo can find it
install(TARGETS mrsim_peg_pose_ign_plugin
  DESTINATION lib/${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}
)

# Install models and worlds
install(DIRECTORY models
  DESTINATION share/${PROJECT_NAME}
)
install(DIRECTORY worlds
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY launch 
  DESTINATION share/${PROJECT_NAME})

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME})

# Export targets
ament_export_targets(${PROJECT_NAME} HAS_LIBRARY_TARGET)

# Mark as ament package
ament_package()