#SIMULATION OF DRONE

from acados_template import AcadosSim, AcadosSimSolver
from drone_model import export_quadrotor_ode_model
from common import *
#from utils import plot_pendulum
import numpy as np

def main():

    sim = AcadosSim()
    sim.model = export_quadrotor_ode_model()

    t_sim=2    # [s] tempo di simulazione
    Tf = 0.001    # [s] tempo di integrazione
    nx = sim.model.x.rows()
    nx_eul=nx-1 #RPY instead of quaternion
    N_sim = np.int64(t_sim/Tf)

    # set simulation time
    sim.solver_options.T = Tf
    #set solver options
    sim.solver_options.integrator_type = 'ERK'
    sim.solver_options.num_stages = 4
    sim.solver_options.num_steps = 1

    # create
    acados_integrator = AcadosSimSolver(sim)
    #initial conditions
    z0=2.0
    p0=np.array([0.0, 0.0, z0])
    rpy0=np.array([0.0,0.0,0.0])
    q0=RPY_to_quat(*rpy0)
    v0=np.array([0.0, 0.0, 0.0])
    w0=np.array([0.0,0.0,0.0])
    #initial state
    x0 = np.array([*p0, *v0, *q0, *w0]) #x=[p v q w]
    x0_eul=np.array([*p0, *v0, *rpy0, *w0])   #x=[p v rpy w]
    #initial control value
    hover_thrust = m * g0  # g0 = 9.81 m/s²
    u0 = np.array([0.0, 0.0, hover_thrust, 0.0, 0.0, 0.0])
    #u0=np.zeros(6)

    #only if IRK
    #xdot_init = np.zeros((nx,))

    simX = np.zeros((N_sim+1, nx))
    simX_eul = np.zeros((N_sim+1, nx_eul))
    simX[0,:] = x0
    simX_eul[0,:] = x0_eul

    p   = np.zeros((N_sim+1,3))
    p[0,:] = p0
    v   = np.zeros((N_sim+1,3))
    v[0,:] = v0
    q   = np.zeros((N_sim+1,4))
    q[0,:] = q0
    rpy = np.zeros((N_sim+1,3))
    rpy[0,:] = rpy0 
    w   = np.zeros((N_sim+1,3))
    w0  = w0

    
    #Simulation
    for i in range(N_sim):
        # Note that xdot is only used if an IRK integrator is used
        simX[i+1,:] = acados_integrator.simulate(x=simX[i,:], u=u0)
        
        #saving into state variables
        p[i+1,:]=simX[i+1,0:3]
        v[i+1,:]=simX[i+1,3:6]
        q[i+1,:]=simX[i+1,6:10]
        rpy[i+1,:]=quat_to_RPY(q[i+1,:])
        w[i+1,:]=simX[i+1,10:]


    #forward sensitivity matrix: è la derivata di f (ovvero x_i+1) rispetto ad x ed u
    #dice quanto lo stato prossimo è sensibile a variazioni dello stato o dell'ingresso
    #S_forw = acados_integrator.get("S_forw")
    #print("S_forw, sensitivities of simulation result wrt x,u:\n", S_forw)

    x=ca.horzcat(p,v,rpy,w)



if __name__ == "__main__":
    main()
