- folder for docker configuration (dockerfile, docker run and docker connect)
- Folder for OCP/MPC with AcadosSimulator
- Folder for OCP/MPC with Gazebo Simulator

Notes:
- Hierarchical PID controller still not working correctly
- With OCP planner a reference change from pi/2 to pi still sometimes is taken as longest distance (pi/2 -> -pi)
- Still is needed to interpolate human reference with dynamical planner reference in MPC mode 
