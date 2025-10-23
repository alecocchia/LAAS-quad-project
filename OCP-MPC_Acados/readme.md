# OCP and MPC with ACADOS solver and simulator
This folder contains a OCP/MPC framework used to track an object trajectory while satisfying model constraints and optimizing some quantities.
This framework uses ACADOS both to solve the optimization problem defined in OCP/MPC and to simulate drone dynamics.
How to use: 
  Run in Python OCP_main.py or MPC_main.py. Animation and plots of simulation will be shown.

What you can easily change in main scripts:
  - references (mutual distance[m], azimuth angle[rad] and polar angle[rad]) between object and drone, expressed with respect to drone. 
    So, for example, dist=2 means that the desired distance between object and drone is 2 meters.
  - weights, to give more or less importance to cost function terms
