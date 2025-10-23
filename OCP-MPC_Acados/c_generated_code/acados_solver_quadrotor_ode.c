/*
 * Copyright (c) The acados authors.
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */

// standard
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
// acados
// #include "acados/utils/print.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"

// example specific

#include "quadrotor_ode_model/quadrotor_ode_model.h"


#include "quadrotor_ode_cost/quadrotor_ode_cost.h"



#include "acados_solver_quadrotor_ode.h"

#define NX     QUADROTOR_ODE_NX
#define NZ     QUADROTOR_ODE_NZ
#define NU     QUADROTOR_ODE_NU
#define NP     QUADROTOR_ODE_NP
#define NP_GLOBAL     QUADROTOR_ODE_NP_GLOBAL
#define NY0    QUADROTOR_ODE_NY0
#define NY     QUADROTOR_ODE_NY
#define NYN    QUADROTOR_ODE_NYN

#define NBX    QUADROTOR_ODE_NBX
#define NBX0   QUADROTOR_ODE_NBX0
#define NBU    QUADROTOR_ODE_NBU
#define NG     QUADROTOR_ODE_NG
#define NBXN   QUADROTOR_ODE_NBXN
#define NGN    QUADROTOR_ODE_NGN

#define NH     QUADROTOR_ODE_NH
#define NHN    QUADROTOR_ODE_NHN
#define NH0    QUADROTOR_ODE_NH0
#define NPHI   QUADROTOR_ODE_NPHI
#define NPHIN  QUADROTOR_ODE_NPHIN
#define NPHI0  QUADROTOR_ODE_NPHI0
#define NR     QUADROTOR_ODE_NR

#define NS     QUADROTOR_ODE_NS
#define NS0    QUADROTOR_ODE_NS0
#define NSN    QUADROTOR_ODE_NSN

#define NSBX   QUADROTOR_ODE_NSBX
#define NSBU   QUADROTOR_ODE_NSBU
#define NSH0   QUADROTOR_ODE_NSH0
#define NSH    QUADROTOR_ODE_NSH
#define NSHN   QUADROTOR_ODE_NSHN
#define NSG    QUADROTOR_ODE_NSG
#define NSPHI0 QUADROTOR_ODE_NSPHI0
#define NSPHI  QUADROTOR_ODE_NSPHI
#define NSPHIN QUADROTOR_ODE_NSPHIN
#define NSGN   QUADROTOR_ODE_NSGN
#define NSBXN  QUADROTOR_ODE_NSBXN



// ** solver data **

quadrotor_ode_solver_capsule * quadrotor_ode_acados_create_capsule(void)
{
    void* capsule_mem = malloc(sizeof(quadrotor_ode_solver_capsule));
    quadrotor_ode_solver_capsule *capsule = (quadrotor_ode_solver_capsule *) capsule_mem;

    return capsule;
}


int quadrotor_ode_acados_free_capsule(quadrotor_ode_solver_capsule *capsule)
{
    free(capsule);
    return 0;
}


int quadrotor_ode_acados_create(quadrotor_ode_solver_capsule* capsule)
{
    int N_shooting_intervals = QUADROTOR_ODE_N;
    double* new_time_steps = NULL; // NULL -> don't alter the code generated time-steps
    return quadrotor_ode_acados_create_with_discretization(capsule, N_shooting_intervals, new_time_steps);
}


int quadrotor_ode_acados_update_time_steps(quadrotor_ode_solver_capsule* capsule, int N, double* new_time_steps)
{

    if (N != capsule->nlp_solver_plan->N) {
        fprintf(stderr, "quadrotor_ode_acados_update_time_steps: given number of time steps (= %d) " \
            "differs from the currently allocated number of " \
            "time steps (= %d)!\n" \
            "Please recreate with new discretization and provide a new vector of time_stamps!\n",
            N, capsule->nlp_solver_plan->N);
        return 1;
    }

    ocp_nlp_config * nlp_config = capsule->nlp_config;
    ocp_nlp_dims * nlp_dims = capsule->nlp_dims;
    ocp_nlp_in * nlp_in = capsule->nlp_in;

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_in_set(nlp_config, nlp_dims, nlp_in, i, "Ts", &new_time_steps[i]);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "scaling", &new_time_steps[i]);
    }
    return 0;

}

/**
 * Internal function for quadrotor_ode_acados_create: step 1
 */
void quadrotor_ode_acados_create_set_plan(ocp_nlp_plan_t* nlp_solver_plan, const int N)
{
    assert(N == nlp_solver_plan->N);

    /************************************************
    *  plan
    ************************************************/

    nlp_solver_plan->nlp_solver = SQP;

    nlp_solver_plan->ocp_qp_solver_plan.qp_solver = PARTIAL_CONDENSING_HPIPM;
    nlp_solver_plan->relaxed_ocp_qp_solver_plan.qp_solver = PARTIAL_CONDENSING_HPIPM;
    nlp_solver_plan->nlp_cost[0] = NONLINEAR_LS;
    for (int i = 1; i < N; i++)
        nlp_solver_plan->nlp_cost[i] = NONLINEAR_LS;

    nlp_solver_plan->nlp_cost[N] = NONLINEAR_LS;

    for (int i = 0; i < N; i++)
    {
        nlp_solver_plan->nlp_dynamics[i] = CONTINUOUS_MODEL;
        nlp_solver_plan->sim_solver_plan[i].sim_solver = ERK;
    }

    nlp_solver_plan->nlp_constraints[0] = BGH;

    for (int i = 1; i < N; i++)
    {
        nlp_solver_plan->nlp_constraints[i] = BGH;
    }
    nlp_solver_plan->nlp_constraints[N] = BGH;

    nlp_solver_plan->regularization = NO_REGULARIZE;

    nlp_solver_plan->globalization = MERIT_BACKTRACKING;
}


static ocp_nlp_dims* quadrotor_ode_acados_create_setup_dimensions(quadrotor_ode_solver_capsule* capsule)
{
    ocp_nlp_plan_t* nlp_solver_plan = capsule->nlp_solver_plan;
    const int N = nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;

    /************************************************
    *  dimensions
    ************************************************/
    #define NINTNP1MEMS 18
    int* intNp1mem = (int*)malloc( (N+1)*sizeof(int)*NINTNP1MEMS );

    int* nx    = intNp1mem + (N+1)*0;
    int* nu    = intNp1mem + (N+1)*1;
    int* nbx   = intNp1mem + (N+1)*2;
    int* nbu   = intNp1mem + (N+1)*3;
    int* nsbx  = intNp1mem + (N+1)*4;
    int* nsbu  = intNp1mem + (N+1)*5;
    int* nsg   = intNp1mem + (N+1)*6;
    int* nsh   = intNp1mem + (N+1)*7;
    int* nsphi = intNp1mem + (N+1)*8;
    int* ns    = intNp1mem + (N+1)*9;
    int* ng    = intNp1mem + (N+1)*10;
    int* nh    = intNp1mem + (N+1)*11;
    int* nphi  = intNp1mem + (N+1)*12;
    int* nz    = intNp1mem + (N+1)*13;
    int* ny    = intNp1mem + (N+1)*14;
    int* nr    = intNp1mem + (N+1)*15;
    int* nbxe  = intNp1mem + (N+1)*16;
    int* np  = intNp1mem + (N+1)*17;

    for (int i = 0; i < N+1; i++)
    {
        // common
        nx[i]     = NX;
        nu[i]     = NU;
        nz[i]     = NZ;
        ns[i]     = NS;
        // cost
        ny[i]     = NY;
        // constraints
        nbx[i]    = NBX;
        nbu[i]    = NBU;
        nsbx[i]   = NSBX;
        nsbu[i]   = NSBU;
        nsg[i]    = NSG;
        nsh[i]    = NSH;
        nsphi[i]  = NSPHI;
        ng[i]     = NG;
        nh[i]     = NH;
        nphi[i]   = NPHI;
        nr[i]     = NR;
        nbxe[i]   = 0;
        np[i]     = NP;
    }

    // for initial state
    nbx[0] = NBX0;
    nsbx[0] = 0;
    ns[0] = NS0;
    
    nbxe[0] = 13;
    
    ny[0] = NY0;
    nh[0] = NH0;
    nsh[0] = NSH0;
    nsphi[0] = NSPHI0;
    nphi[0] = NPHI0;


    // terminal - common
    nu[N]   = 0;
    nz[N]   = 0;
    ns[N]   = NSN;
    // cost
    ny[N]   = NYN;
    // constraint
    nbx[N]   = NBXN;
    nbu[N]   = 0;
    ng[N]    = NGN;
    nh[N]    = NHN;
    nphi[N]  = NPHIN;
    nr[N]    = 0;

    nsbx[N]  = NSBXN;
    nsbu[N]  = 0;
    nsg[N]   = NSGN;
    nsh[N]   = NSHN;
    nsphi[N] = NSPHIN;

    /* create and set ocp_nlp_dims */
    ocp_nlp_dims * nlp_dims = ocp_nlp_dims_create(nlp_config);

    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nx", nx);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nu", nu);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nz", nz);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "ns", ns);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "np", np);

    ocp_nlp_dims_set_global(nlp_config, nlp_dims, "np_global", 0);
    ocp_nlp_dims_set_global(nlp_config, nlp_dims, "n_global_data", 0);

    for (int i = 0; i <= N; i++)
    {
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbx", &nbx[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbu", &nbu[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsbx", &nsbx[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsbu", &nsbu[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "ng", &ng[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsg", &nsg[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbxe", &nbxe[i]);
    }
    ocp_nlp_dims_set_cost(nlp_config, nlp_dims, 0, "ny", &ny[0]);
    for (int i = 1; i < N; i++)
        ocp_nlp_dims_set_cost(nlp_config, nlp_dims, i, "ny", &ny[i]);
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, 0, "nh", &nh[0]);
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, 0, "nsh", &nsh[0]);

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nh", &nh[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsh", &nsh[i]);
    }
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, N, "nh", &nh[N]);
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, N, "nsh", &nsh[N]);
    ocp_nlp_dims_set_cost(nlp_config, nlp_dims, N, "ny", &ny[N]);
    free(intNp1mem);

    return nlp_dims;
}


/**
 * Internal function for quadrotor_ode_acados_create: step 3
 */
void quadrotor_ode_acados_create_setup_functions(quadrotor_ode_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;

    /************************************************
    *  external functions
    ************************************************/

#define MAP_CASADI_FNC(__CAPSULE_FNC__, __MODEL_BASE_FNC__) do{ \
        capsule->__CAPSULE_FNC__.casadi_fun = & __MODEL_BASE_FNC__ ;\
        capsule->__CAPSULE_FNC__.casadi_n_in = & __MODEL_BASE_FNC__ ## _n_in; \
        capsule->__CAPSULE_FNC__.casadi_n_out = & __MODEL_BASE_FNC__ ## _n_out; \
        capsule->__CAPSULE_FNC__.casadi_sparsity_in = & __MODEL_BASE_FNC__ ## _sparsity_in; \
        capsule->__CAPSULE_FNC__.casadi_sparsity_out = & __MODEL_BASE_FNC__ ## _sparsity_out; \
        capsule->__CAPSULE_FNC__.casadi_work = & __MODEL_BASE_FNC__ ## _work; \
        external_function_external_param_casadi_create(&capsule->__CAPSULE_FNC__, &ext_fun_opts); \
    } while(false)

    external_function_opts ext_fun_opts;
    external_function_opts_set_to_default(&ext_fun_opts);


    ext_fun_opts.external_workspace = true;
    if (N > 0)
    {
        // nonlinear least squares function
        MAP_CASADI_FNC(cost_y_0_fun, quadrotor_ode_cost_y_0_fun);
        MAP_CASADI_FNC(cost_y_0_fun_jac_ut_xt, quadrotor_ode_cost_y_0_fun_jac_ut_xt);



    
        // explicit ode
        capsule->expl_vde_forw = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*N);
        for (int i = 0; i < N; i++) {
            MAP_CASADI_FNC(expl_vde_forw[i], quadrotor_ode_expl_vde_forw);
        }

        capsule->expl_ode_fun = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*N);
        for (int i = 0; i < N; i++) {
            MAP_CASADI_FNC(expl_ode_fun[i], quadrotor_ode_expl_ode_fun);
        }

        capsule->expl_vde_adj = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*N);
        for (int i = 0; i < N; i++) {
            MAP_CASADI_FNC(expl_vde_adj[i], quadrotor_ode_expl_vde_adj);
        }

    
        // nonlinear least squares cost
        capsule->cost_y_fun = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*(N-1));
        for (int i = 0; i < N-1; i++)
        {
            MAP_CASADI_FNC(cost_y_fun[i], quadrotor_ode_cost_y_fun);
        }

        capsule->cost_y_fun_jac_ut_xt = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*(N-1));
        for (int i = 0; i < N-1; i++)
        {
            MAP_CASADI_FNC(cost_y_fun_jac_ut_xt[i], quadrotor_ode_cost_y_fun_jac_ut_xt);
        }
    } // N > 0
    // nonlinear least square function
    MAP_CASADI_FNC(cost_y_e_fun, quadrotor_ode_cost_y_e_fun);
    MAP_CASADI_FNC(cost_y_e_fun_jac_ut_xt, quadrotor_ode_cost_y_e_fun_jac_ut_xt);

#undef MAP_CASADI_FNC
}


/**
 * Internal function for quadrotor_ode_acados_create: step 5
 */
void quadrotor_ode_acados_create_set_default_parameters(quadrotor_ode_solver_capsule* capsule)
{

    const int N = capsule->nlp_solver_plan->N;
    // initialize parameters to nominal value
    double* p = calloc(NP, sizeof(double));
    p[0] = 2;
    p[1] = 2;
    p[8] = 1.5707963267948966;

    for (int i = 0; i <= N; i++) {
        quadrotor_ode_acados_update_params(capsule, i, p, NP);
    }
    free(p);


    // no global parameters defined
}


/**
 * Internal function for quadrotor_ode_acados_create: step 5
 */
void quadrotor_ode_acados_setup_nlp_in(quadrotor_ode_solver_capsule* capsule, const int N, double* new_time_steps)
{
    assert(N == capsule->nlp_solver_plan->N);
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;

    int tmp_int = 0;

    /************************************************
    *  nlp_in
    ************************************************/
    ocp_nlp_in * nlp_in = capsule->nlp_in;
    /************************************************
    *  nlp_out
    ************************************************/
    ocp_nlp_out * nlp_out = capsule->nlp_out;

    // set up time_steps and cost_scaling

    if (new_time_steps)
    {
        // NOTE: this sets scaling and time_steps
        quadrotor_ode_acados_update_time_steps(capsule, N, new_time_steps);
    }
    else
    {
        // set time_steps
    
        double time_step = 0.01;
        for (int i = 0; i < N; i++)
        {
            ocp_nlp_in_set(nlp_config, nlp_dims, nlp_in, i, "Ts", &time_step);
        }
        // set cost scaling
        double* cost_scaling = malloc((N+1)*sizeof(double));
        cost_scaling[0] = 0.01;
        cost_scaling[1] = 0.01;
        cost_scaling[2] = 0.01;
        cost_scaling[3] = 0.01;
        cost_scaling[4] = 0.01;
        cost_scaling[5] = 0.01;
        cost_scaling[6] = 0.01;
        cost_scaling[7] = 0.01;
        cost_scaling[8] = 0.01;
        cost_scaling[9] = 0.01;
        cost_scaling[10] = 0.01;
        cost_scaling[11] = 0.01;
        cost_scaling[12] = 0.01;
        cost_scaling[13] = 0.01;
        cost_scaling[14] = 0.01;
        cost_scaling[15] = 0.01;
        cost_scaling[16] = 0.01;
        cost_scaling[17] = 0.01;
        cost_scaling[18] = 0.01;
        cost_scaling[19] = 0.01;
        cost_scaling[20] = 0.01;
        cost_scaling[21] = 0.01;
        cost_scaling[22] = 0.01;
        cost_scaling[23] = 0.01;
        cost_scaling[24] = 0.01;
        cost_scaling[25] = 0.01;
        cost_scaling[26] = 0.01;
        cost_scaling[27] = 0.01;
        cost_scaling[28] = 0.01;
        cost_scaling[29] = 0.01;
        cost_scaling[30] = 0.01;
        cost_scaling[31] = 0.01;
        cost_scaling[32] = 0.01;
        cost_scaling[33] = 0.01;
        cost_scaling[34] = 0.01;
        cost_scaling[35] = 0.01;
        cost_scaling[36] = 0.01;
        cost_scaling[37] = 0.01;
        cost_scaling[38] = 0.01;
        cost_scaling[39] = 0.01;
        cost_scaling[40] = 0.01;
        cost_scaling[41] = 0.01;
        cost_scaling[42] = 0.01;
        cost_scaling[43] = 0.01;
        cost_scaling[44] = 0.01;
        cost_scaling[45] = 0.01;
        cost_scaling[46] = 0.01;
        cost_scaling[47] = 0.01;
        cost_scaling[48] = 0.01;
        cost_scaling[49] = 0.01;
        cost_scaling[50] = 0.01;
        cost_scaling[51] = 0.01;
        cost_scaling[52] = 0.01;
        cost_scaling[53] = 0.01;
        cost_scaling[54] = 0.01;
        cost_scaling[55] = 0.01;
        cost_scaling[56] = 0.01;
        cost_scaling[57] = 0.01;
        cost_scaling[58] = 0.01;
        cost_scaling[59] = 0.01;
        cost_scaling[60] = 0.01;
        cost_scaling[61] = 0.01;
        cost_scaling[62] = 0.01;
        cost_scaling[63] = 0.01;
        cost_scaling[64] = 0.01;
        cost_scaling[65] = 0.01;
        cost_scaling[66] = 0.01;
        cost_scaling[67] = 0.01;
        cost_scaling[68] = 0.01;
        cost_scaling[69] = 0.01;
        cost_scaling[70] = 0.01;
        cost_scaling[71] = 0.01;
        cost_scaling[72] = 0.01;
        cost_scaling[73] = 0.01;
        cost_scaling[74] = 0.01;
        cost_scaling[75] = 0.01;
        cost_scaling[76] = 0.01;
        cost_scaling[77] = 0.01;
        cost_scaling[78] = 0.01;
        cost_scaling[79] = 0.01;
        cost_scaling[80] = 0.01;
        cost_scaling[81] = 0.01;
        cost_scaling[82] = 0.01;
        cost_scaling[83] = 0.01;
        cost_scaling[84] = 0.01;
        cost_scaling[85] = 0.01;
        cost_scaling[86] = 0.01;
        cost_scaling[87] = 0.01;
        cost_scaling[88] = 0.01;
        cost_scaling[89] = 0.01;
        cost_scaling[90] = 0.01;
        cost_scaling[91] = 0.01;
        cost_scaling[92] = 0.01;
        cost_scaling[93] = 0.01;
        cost_scaling[94] = 0.01;
        cost_scaling[95] = 0.01;
        cost_scaling[96] = 0.01;
        cost_scaling[97] = 0.01;
        cost_scaling[98] = 0.01;
        cost_scaling[99] = 0.01;
        cost_scaling[100] = 0.01;
        cost_scaling[101] = 0.01;
        cost_scaling[102] = 0.01;
        cost_scaling[103] = 0.01;
        cost_scaling[104] = 0.01;
        cost_scaling[105] = 0.01;
        cost_scaling[106] = 0.01;
        cost_scaling[107] = 0.01;
        cost_scaling[108] = 0.01;
        cost_scaling[109] = 0.01;
        cost_scaling[110] = 0.01;
        cost_scaling[111] = 0.01;
        cost_scaling[112] = 0.01;
        cost_scaling[113] = 0.01;
        cost_scaling[114] = 0.01;
        cost_scaling[115] = 0.01;
        cost_scaling[116] = 0.01;
        cost_scaling[117] = 0.01;
        cost_scaling[118] = 0.01;
        cost_scaling[119] = 0.01;
        cost_scaling[120] = 0.01;
        cost_scaling[121] = 0.01;
        cost_scaling[122] = 0.01;
        cost_scaling[123] = 0.01;
        cost_scaling[124] = 0.01;
        cost_scaling[125] = 0.01;
        cost_scaling[126] = 0.01;
        cost_scaling[127] = 0.01;
        cost_scaling[128] = 0.01;
        cost_scaling[129] = 0.01;
        cost_scaling[130] = 0.01;
        cost_scaling[131] = 0.01;
        cost_scaling[132] = 0.01;
        cost_scaling[133] = 0.01;
        cost_scaling[134] = 0.01;
        cost_scaling[135] = 0.01;
        cost_scaling[136] = 0.01;
        cost_scaling[137] = 0.01;
        cost_scaling[138] = 0.01;
        cost_scaling[139] = 0.01;
        cost_scaling[140] = 0.01;
        cost_scaling[141] = 0.01;
        cost_scaling[142] = 0.01;
        cost_scaling[143] = 0.01;
        cost_scaling[144] = 0.01;
        cost_scaling[145] = 0.01;
        cost_scaling[146] = 0.01;
        cost_scaling[147] = 0.01;
        cost_scaling[148] = 0.01;
        cost_scaling[149] = 0.01;
        cost_scaling[150] = 0.01;
        cost_scaling[151] = 0.01;
        cost_scaling[152] = 0.01;
        cost_scaling[153] = 0.01;
        cost_scaling[154] = 0.01;
        cost_scaling[155] = 0.01;
        cost_scaling[156] = 0.01;
        cost_scaling[157] = 0.01;
        cost_scaling[158] = 0.01;
        cost_scaling[159] = 0.01;
        cost_scaling[160] = 0.01;
        cost_scaling[161] = 0.01;
        cost_scaling[162] = 0.01;
        cost_scaling[163] = 0.01;
        cost_scaling[164] = 0.01;
        cost_scaling[165] = 0.01;
        cost_scaling[166] = 0.01;
        cost_scaling[167] = 0.01;
        cost_scaling[168] = 0.01;
        cost_scaling[169] = 0.01;
        cost_scaling[170] = 0.01;
        cost_scaling[171] = 0.01;
        cost_scaling[172] = 0.01;
        cost_scaling[173] = 0.01;
        cost_scaling[174] = 0.01;
        cost_scaling[175] = 0.01;
        cost_scaling[176] = 0.01;
        cost_scaling[177] = 0.01;
        cost_scaling[178] = 0.01;
        cost_scaling[179] = 0.01;
        cost_scaling[180] = 0.01;
        cost_scaling[181] = 0.01;
        cost_scaling[182] = 0.01;
        cost_scaling[183] = 0.01;
        cost_scaling[184] = 0.01;
        cost_scaling[185] = 0.01;
        cost_scaling[186] = 0.01;
        cost_scaling[187] = 0.01;
        cost_scaling[188] = 0.01;
        cost_scaling[189] = 0.01;
        cost_scaling[190] = 0.01;
        cost_scaling[191] = 0.01;
        cost_scaling[192] = 0.01;
        cost_scaling[193] = 0.01;
        cost_scaling[194] = 0.01;
        cost_scaling[195] = 0.01;
        cost_scaling[196] = 0.01;
        cost_scaling[197] = 0.01;
        cost_scaling[198] = 0.01;
        cost_scaling[199] = 0.01;
        cost_scaling[200] = 0.01;
        cost_scaling[201] = 0.01;
        cost_scaling[202] = 0.01;
        cost_scaling[203] = 0.01;
        cost_scaling[204] = 0.01;
        cost_scaling[205] = 0.01;
        cost_scaling[206] = 0.01;
        cost_scaling[207] = 0.01;
        cost_scaling[208] = 0.01;
        cost_scaling[209] = 0.01;
        cost_scaling[210] = 0.01;
        cost_scaling[211] = 0.01;
        cost_scaling[212] = 0.01;
        cost_scaling[213] = 0.01;
        cost_scaling[214] = 0.01;
        cost_scaling[215] = 0.01;
        cost_scaling[216] = 0.01;
        cost_scaling[217] = 0.01;
        cost_scaling[218] = 0.01;
        cost_scaling[219] = 0.01;
        cost_scaling[220] = 0.01;
        cost_scaling[221] = 0.01;
        cost_scaling[222] = 0.01;
        cost_scaling[223] = 0.01;
        cost_scaling[224] = 0.01;
        cost_scaling[225] = 0.01;
        cost_scaling[226] = 0.01;
        cost_scaling[227] = 0.01;
        cost_scaling[228] = 0.01;
        cost_scaling[229] = 0.01;
        cost_scaling[230] = 0.01;
        cost_scaling[231] = 0.01;
        cost_scaling[232] = 0.01;
        cost_scaling[233] = 0.01;
        cost_scaling[234] = 0.01;
        cost_scaling[235] = 0.01;
        cost_scaling[236] = 0.01;
        cost_scaling[237] = 0.01;
        cost_scaling[238] = 0.01;
        cost_scaling[239] = 0.01;
        cost_scaling[240] = 0.01;
        cost_scaling[241] = 0.01;
        cost_scaling[242] = 0.01;
        cost_scaling[243] = 0.01;
        cost_scaling[244] = 0.01;
        cost_scaling[245] = 0.01;
        cost_scaling[246] = 0.01;
        cost_scaling[247] = 0.01;
        cost_scaling[248] = 0.01;
        cost_scaling[249] = 0.01;
        cost_scaling[250] = 0.01;
        cost_scaling[251] = 0.01;
        cost_scaling[252] = 0.01;
        cost_scaling[253] = 0.01;
        cost_scaling[254] = 0.01;
        cost_scaling[255] = 0.01;
        cost_scaling[256] = 0.01;
        cost_scaling[257] = 0.01;
        cost_scaling[258] = 0.01;
        cost_scaling[259] = 0.01;
        cost_scaling[260] = 0.01;
        cost_scaling[261] = 0.01;
        cost_scaling[262] = 0.01;
        cost_scaling[263] = 0.01;
        cost_scaling[264] = 0.01;
        cost_scaling[265] = 0.01;
        cost_scaling[266] = 0.01;
        cost_scaling[267] = 0.01;
        cost_scaling[268] = 0.01;
        cost_scaling[269] = 0.01;
        cost_scaling[270] = 0.01;
        cost_scaling[271] = 0.01;
        cost_scaling[272] = 0.01;
        cost_scaling[273] = 0.01;
        cost_scaling[274] = 0.01;
        cost_scaling[275] = 0.01;
        cost_scaling[276] = 0.01;
        cost_scaling[277] = 0.01;
        cost_scaling[278] = 0.01;
        cost_scaling[279] = 0.01;
        cost_scaling[280] = 0.01;
        cost_scaling[281] = 0.01;
        cost_scaling[282] = 0.01;
        cost_scaling[283] = 0.01;
        cost_scaling[284] = 0.01;
        cost_scaling[285] = 0.01;
        cost_scaling[286] = 0.01;
        cost_scaling[287] = 0.01;
        cost_scaling[288] = 0.01;
        cost_scaling[289] = 0.01;
        cost_scaling[290] = 0.01;
        cost_scaling[291] = 0.01;
        cost_scaling[292] = 0.01;
        cost_scaling[293] = 0.01;
        cost_scaling[294] = 0.01;
        cost_scaling[295] = 0.01;
        cost_scaling[296] = 0.01;
        cost_scaling[297] = 0.01;
        cost_scaling[298] = 0.01;
        cost_scaling[299] = 0.01;
        cost_scaling[300] = 0.01;
        cost_scaling[301] = 0.01;
        cost_scaling[302] = 0.01;
        cost_scaling[303] = 0.01;
        cost_scaling[304] = 0.01;
        cost_scaling[305] = 0.01;
        cost_scaling[306] = 0.01;
        cost_scaling[307] = 0.01;
        cost_scaling[308] = 0.01;
        cost_scaling[309] = 0.01;
        cost_scaling[310] = 0.01;
        cost_scaling[311] = 0.01;
        cost_scaling[312] = 0.01;
        cost_scaling[313] = 0.01;
        cost_scaling[314] = 0.01;
        cost_scaling[315] = 0.01;
        cost_scaling[316] = 0.01;
        cost_scaling[317] = 0.01;
        cost_scaling[318] = 0.01;
        cost_scaling[319] = 0.01;
        cost_scaling[320] = 0.01;
        cost_scaling[321] = 0.01;
        cost_scaling[322] = 0.01;
        cost_scaling[323] = 0.01;
        cost_scaling[324] = 0.01;
        cost_scaling[325] = 0.01;
        cost_scaling[326] = 0.01;
        cost_scaling[327] = 0.01;
        cost_scaling[328] = 0.01;
        cost_scaling[329] = 0.01;
        cost_scaling[330] = 0.01;
        cost_scaling[331] = 0.01;
        cost_scaling[332] = 0.01;
        cost_scaling[333] = 0.01;
        cost_scaling[334] = 0.01;
        cost_scaling[335] = 0.01;
        cost_scaling[336] = 0.01;
        cost_scaling[337] = 0.01;
        cost_scaling[338] = 0.01;
        cost_scaling[339] = 0.01;
        cost_scaling[340] = 0.01;
        cost_scaling[341] = 0.01;
        cost_scaling[342] = 0.01;
        cost_scaling[343] = 0.01;
        cost_scaling[344] = 0.01;
        cost_scaling[345] = 0.01;
        cost_scaling[346] = 0.01;
        cost_scaling[347] = 0.01;
        cost_scaling[348] = 0.01;
        cost_scaling[349] = 0.01;
        cost_scaling[350] = 0.01;
        cost_scaling[351] = 0.01;
        cost_scaling[352] = 0.01;
        cost_scaling[353] = 0.01;
        cost_scaling[354] = 0.01;
        cost_scaling[355] = 0.01;
        cost_scaling[356] = 0.01;
        cost_scaling[357] = 0.01;
        cost_scaling[358] = 0.01;
        cost_scaling[359] = 0.01;
        cost_scaling[360] = 0.01;
        cost_scaling[361] = 0.01;
        cost_scaling[362] = 0.01;
        cost_scaling[363] = 0.01;
        cost_scaling[364] = 0.01;
        cost_scaling[365] = 0.01;
        cost_scaling[366] = 0.01;
        cost_scaling[367] = 0.01;
        cost_scaling[368] = 0.01;
        cost_scaling[369] = 0.01;
        cost_scaling[370] = 0.01;
        cost_scaling[371] = 0.01;
        cost_scaling[372] = 0.01;
        cost_scaling[373] = 0.01;
        cost_scaling[374] = 0.01;
        cost_scaling[375] = 0.01;
        cost_scaling[376] = 0.01;
        cost_scaling[377] = 0.01;
        cost_scaling[378] = 0.01;
        cost_scaling[379] = 0.01;
        cost_scaling[380] = 0.01;
        cost_scaling[381] = 0.01;
        cost_scaling[382] = 0.01;
        cost_scaling[383] = 0.01;
        cost_scaling[384] = 0.01;
        cost_scaling[385] = 0.01;
        cost_scaling[386] = 0.01;
        cost_scaling[387] = 0.01;
        cost_scaling[388] = 0.01;
        cost_scaling[389] = 0.01;
        cost_scaling[390] = 0.01;
        cost_scaling[391] = 0.01;
        cost_scaling[392] = 0.01;
        cost_scaling[393] = 0.01;
        cost_scaling[394] = 0.01;
        cost_scaling[395] = 0.01;
        cost_scaling[396] = 0.01;
        cost_scaling[397] = 0.01;
        cost_scaling[398] = 0.01;
        cost_scaling[399] = 0.01;
        cost_scaling[400] = 0.01;
        cost_scaling[401] = 0.01;
        cost_scaling[402] = 0.01;
        cost_scaling[403] = 0.01;
        cost_scaling[404] = 0.01;
        cost_scaling[405] = 0.01;
        cost_scaling[406] = 0.01;
        cost_scaling[407] = 0.01;
        cost_scaling[408] = 0.01;
        cost_scaling[409] = 0.01;
        cost_scaling[410] = 0.01;
        cost_scaling[411] = 0.01;
        cost_scaling[412] = 0.01;
        cost_scaling[413] = 0.01;
        cost_scaling[414] = 0.01;
        cost_scaling[415] = 0.01;
        cost_scaling[416] = 0.01;
        cost_scaling[417] = 0.01;
        cost_scaling[418] = 0.01;
        cost_scaling[419] = 0.01;
        cost_scaling[420] = 0.01;
        cost_scaling[421] = 0.01;
        cost_scaling[422] = 0.01;
        cost_scaling[423] = 0.01;
        cost_scaling[424] = 0.01;
        cost_scaling[425] = 0.01;
        cost_scaling[426] = 0.01;
        cost_scaling[427] = 0.01;
        cost_scaling[428] = 0.01;
        cost_scaling[429] = 0.01;
        cost_scaling[430] = 0.01;
        cost_scaling[431] = 0.01;
        cost_scaling[432] = 0.01;
        cost_scaling[433] = 0.01;
        cost_scaling[434] = 0.01;
        cost_scaling[435] = 0.01;
        cost_scaling[436] = 0.01;
        cost_scaling[437] = 0.01;
        cost_scaling[438] = 0.01;
        cost_scaling[439] = 0.01;
        cost_scaling[440] = 0.01;
        cost_scaling[441] = 0.01;
        cost_scaling[442] = 0.01;
        cost_scaling[443] = 0.01;
        cost_scaling[444] = 0.01;
        cost_scaling[445] = 0.01;
        cost_scaling[446] = 0.01;
        cost_scaling[447] = 0.01;
        cost_scaling[448] = 0.01;
        cost_scaling[449] = 0.01;
        cost_scaling[450] = 0.01;
        cost_scaling[451] = 0.01;
        cost_scaling[452] = 0.01;
        cost_scaling[453] = 0.01;
        cost_scaling[454] = 0.01;
        cost_scaling[455] = 0.01;
        cost_scaling[456] = 0.01;
        cost_scaling[457] = 0.01;
        cost_scaling[458] = 0.01;
        cost_scaling[459] = 0.01;
        cost_scaling[460] = 0.01;
        cost_scaling[461] = 0.01;
        cost_scaling[462] = 0.01;
        cost_scaling[463] = 0.01;
        cost_scaling[464] = 0.01;
        cost_scaling[465] = 0.01;
        cost_scaling[466] = 0.01;
        cost_scaling[467] = 0.01;
        cost_scaling[468] = 0.01;
        cost_scaling[469] = 0.01;
        cost_scaling[470] = 0.01;
        cost_scaling[471] = 0.01;
        cost_scaling[472] = 0.01;
        cost_scaling[473] = 0.01;
        cost_scaling[474] = 0.01;
        cost_scaling[475] = 0.01;
        cost_scaling[476] = 0.01;
        cost_scaling[477] = 0.01;
        cost_scaling[478] = 0.01;
        cost_scaling[479] = 0.01;
        cost_scaling[480] = 0.01;
        cost_scaling[481] = 0.01;
        cost_scaling[482] = 0.01;
        cost_scaling[483] = 0.01;
        cost_scaling[484] = 0.01;
        cost_scaling[485] = 0.01;
        cost_scaling[486] = 0.01;
        cost_scaling[487] = 0.01;
        cost_scaling[488] = 0.01;
        cost_scaling[489] = 0.01;
        cost_scaling[490] = 0.01;
        cost_scaling[491] = 0.01;
        cost_scaling[492] = 0.01;
        cost_scaling[493] = 0.01;
        cost_scaling[494] = 0.01;
        cost_scaling[495] = 0.01;
        cost_scaling[496] = 0.01;
        cost_scaling[497] = 0.01;
        cost_scaling[498] = 0.01;
        cost_scaling[499] = 0.01;
        cost_scaling[500] = 0.01;
        cost_scaling[501] = 0.01;
        cost_scaling[502] = 0.01;
        cost_scaling[503] = 0.01;
        cost_scaling[504] = 0.01;
        cost_scaling[505] = 0.01;
        cost_scaling[506] = 0.01;
        cost_scaling[507] = 0.01;
        cost_scaling[508] = 0.01;
        cost_scaling[509] = 0.01;
        cost_scaling[510] = 0.01;
        cost_scaling[511] = 0.01;
        cost_scaling[512] = 0.01;
        cost_scaling[513] = 0.01;
        cost_scaling[514] = 0.01;
        cost_scaling[515] = 0.01;
        cost_scaling[516] = 0.01;
        cost_scaling[517] = 0.01;
        cost_scaling[518] = 0.01;
        cost_scaling[519] = 0.01;
        cost_scaling[520] = 0.01;
        cost_scaling[521] = 0.01;
        cost_scaling[522] = 0.01;
        cost_scaling[523] = 0.01;
        cost_scaling[524] = 0.01;
        cost_scaling[525] = 0.01;
        cost_scaling[526] = 0.01;
        cost_scaling[527] = 0.01;
        cost_scaling[528] = 0.01;
        cost_scaling[529] = 0.01;
        cost_scaling[530] = 0.01;
        cost_scaling[531] = 0.01;
        cost_scaling[532] = 0.01;
        cost_scaling[533] = 0.01;
        cost_scaling[534] = 0.01;
        cost_scaling[535] = 0.01;
        cost_scaling[536] = 0.01;
        cost_scaling[537] = 0.01;
        cost_scaling[538] = 0.01;
        cost_scaling[539] = 0.01;
        cost_scaling[540] = 0.01;
        cost_scaling[541] = 0.01;
        cost_scaling[542] = 0.01;
        cost_scaling[543] = 0.01;
        cost_scaling[544] = 0.01;
        cost_scaling[545] = 0.01;
        cost_scaling[546] = 0.01;
        cost_scaling[547] = 0.01;
        cost_scaling[548] = 0.01;
        cost_scaling[549] = 0.01;
        cost_scaling[550] = 0.01;
        cost_scaling[551] = 0.01;
        cost_scaling[552] = 0.01;
        cost_scaling[553] = 0.01;
        cost_scaling[554] = 0.01;
        cost_scaling[555] = 0.01;
        cost_scaling[556] = 0.01;
        cost_scaling[557] = 0.01;
        cost_scaling[558] = 0.01;
        cost_scaling[559] = 0.01;
        cost_scaling[560] = 0.01;
        cost_scaling[561] = 0.01;
        cost_scaling[562] = 0.01;
        cost_scaling[563] = 0.01;
        cost_scaling[564] = 0.01;
        cost_scaling[565] = 0.01;
        cost_scaling[566] = 0.01;
        cost_scaling[567] = 0.01;
        cost_scaling[568] = 0.01;
        cost_scaling[569] = 0.01;
        cost_scaling[570] = 0.01;
        cost_scaling[571] = 0.01;
        cost_scaling[572] = 0.01;
        cost_scaling[573] = 0.01;
        cost_scaling[574] = 0.01;
        cost_scaling[575] = 0.01;
        cost_scaling[576] = 0.01;
        cost_scaling[577] = 0.01;
        cost_scaling[578] = 0.01;
        cost_scaling[579] = 0.01;
        cost_scaling[580] = 0.01;
        cost_scaling[581] = 0.01;
        cost_scaling[582] = 0.01;
        cost_scaling[583] = 0.01;
        cost_scaling[584] = 0.01;
        cost_scaling[585] = 0.01;
        cost_scaling[586] = 0.01;
        cost_scaling[587] = 0.01;
        cost_scaling[588] = 0.01;
        cost_scaling[589] = 0.01;
        cost_scaling[590] = 0.01;
        cost_scaling[591] = 0.01;
        cost_scaling[592] = 0.01;
        cost_scaling[593] = 0.01;
        cost_scaling[594] = 0.01;
        cost_scaling[595] = 0.01;
        cost_scaling[596] = 0.01;
        cost_scaling[597] = 0.01;
        cost_scaling[598] = 0.01;
        cost_scaling[599] = 0.01;
        cost_scaling[600] = 0.01;
        cost_scaling[601] = 0.01;
        cost_scaling[602] = 0.01;
        cost_scaling[603] = 0.01;
        cost_scaling[604] = 0.01;
        cost_scaling[605] = 0.01;
        cost_scaling[606] = 0.01;
        cost_scaling[607] = 0.01;
        cost_scaling[608] = 0.01;
        cost_scaling[609] = 0.01;
        cost_scaling[610] = 0.01;
        cost_scaling[611] = 0.01;
        cost_scaling[612] = 0.01;
        cost_scaling[613] = 0.01;
        cost_scaling[614] = 0.01;
        cost_scaling[615] = 0.01;
        cost_scaling[616] = 0.01;
        cost_scaling[617] = 0.01;
        cost_scaling[618] = 0.01;
        cost_scaling[619] = 0.01;
        cost_scaling[620] = 0.01;
        cost_scaling[621] = 0.01;
        cost_scaling[622] = 0.01;
        cost_scaling[623] = 0.01;
        cost_scaling[624] = 0.01;
        cost_scaling[625] = 0.01;
        cost_scaling[626] = 0.01;
        cost_scaling[627] = 0.01;
        cost_scaling[628] = 0.01;
        cost_scaling[629] = 0.01;
        cost_scaling[630] = 0.01;
        cost_scaling[631] = 0.01;
        cost_scaling[632] = 0.01;
        cost_scaling[633] = 0.01;
        cost_scaling[634] = 0.01;
        cost_scaling[635] = 0.01;
        cost_scaling[636] = 0.01;
        cost_scaling[637] = 0.01;
        cost_scaling[638] = 0.01;
        cost_scaling[639] = 0.01;
        cost_scaling[640] = 0.01;
        cost_scaling[641] = 0.01;
        cost_scaling[642] = 0.01;
        cost_scaling[643] = 0.01;
        cost_scaling[644] = 0.01;
        cost_scaling[645] = 0.01;
        cost_scaling[646] = 0.01;
        cost_scaling[647] = 0.01;
        cost_scaling[648] = 0.01;
        cost_scaling[649] = 0.01;
        cost_scaling[650] = 0.01;
        cost_scaling[651] = 0.01;
        cost_scaling[652] = 0.01;
        cost_scaling[653] = 0.01;
        cost_scaling[654] = 0.01;
        cost_scaling[655] = 0.01;
        cost_scaling[656] = 0.01;
        cost_scaling[657] = 0.01;
        cost_scaling[658] = 0.01;
        cost_scaling[659] = 0.01;
        cost_scaling[660] = 0.01;
        cost_scaling[661] = 0.01;
        cost_scaling[662] = 0.01;
        cost_scaling[663] = 0.01;
        cost_scaling[664] = 0.01;
        cost_scaling[665] = 0.01;
        cost_scaling[666] = 0.01;
        cost_scaling[667] = 0.01;
        cost_scaling[668] = 0.01;
        cost_scaling[669] = 0.01;
        cost_scaling[670] = 0.01;
        cost_scaling[671] = 0.01;
        cost_scaling[672] = 0.01;
        cost_scaling[673] = 0.01;
        cost_scaling[674] = 0.01;
        cost_scaling[675] = 0.01;
        cost_scaling[676] = 0.01;
        cost_scaling[677] = 0.01;
        cost_scaling[678] = 0.01;
        cost_scaling[679] = 0.01;
        cost_scaling[680] = 0.01;
        cost_scaling[681] = 0.01;
        cost_scaling[682] = 0.01;
        cost_scaling[683] = 0.01;
        cost_scaling[684] = 0.01;
        cost_scaling[685] = 0.01;
        cost_scaling[686] = 0.01;
        cost_scaling[687] = 0.01;
        cost_scaling[688] = 0.01;
        cost_scaling[689] = 0.01;
        cost_scaling[690] = 0.01;
        cost_scaling[691] = 0.01;
        cost_scaling[692] = 0.01;
        cost_scaling[693] = 0.01;
        cost_scaling[694] = 0.01;
        cost_scaling[695] = 0.01;
        cost_scaling[696] = 0.01;
        cost_scaling[697] = 0.01;
        cost_scaling[698] = 0.01;
        cost_scaling[699] = 0.01;
        cost_scaling[700] = 0.01;
        cost_scaling[701] = 0.01;
        cost_scaling[702] = 0.01;
        cost_scaling[703] = 0.01;
        cost_scaling[704] = 0.01;
        cost_scaling[705] = 0.01;
        cost_scaling[706] = 0.01;
        cost_scaling[707] = 0.01;
        cost_scaling[708] = 0.01;
        cost_scaling[709] = 0.01;
        cost_scaling[710] = 0.01;
        cost_scaling[711] = 0.01;
        cost_scaling[712] = 0.01;
        cost_scaling[713] = 0.01;
        cost_scaling[714] = 0.01;
        cost_scaling[715] = 0.01;
        cost_scaling[716] = 0.01;
        cost_scaling[717] = 0.01;
        cost_scaling[718] = 0.01;
        cost_scaling[719] = 0.01;
        cost_scaling[720] = 0.01;
        cost_scaling[721] = 0.01;
        cost_scaling[722] = 0.01;
        cost_scaling[723] = 0.01;
        cost_scaling[724] = 0.01;
        cost_scaling[725] = 0.01;
        cost_scaling[726] = 0.01;
        cost_scaling[727] = 0.01;
        cost_scaling[728] = 0.01;
        cost_scaling[729] = 0.01;
        cost_scaling[730] = 0.01;
        cost_scaling[731] = 0.01;
        cost_scaling[732] = 0.01;
        cost_scaling[733] = 0.01;
        cost_scaling[734] = 0.01;
        cost_scaling[735] = 0.01;
        cost_scaling[736] = 0.01;
        cost_scaling[737] = 0.01;
        cost_scaling[738] = 0.01;
        cost_scaling[739] = 0.01;
        cost_scaling[740] = 0.01;
        cost_scaling[741] = 0.01;
        cost_scaling[742] = 0.01;
        cost_scaling[743] = 0.01;
        cost_scaling[744] = 0.01;
        cost_scaling[745] = 0.01;
        cost_scaling[746] = 0.01;
        cost_scaling[747] = 0.01;
        cost_scaling[748] = 0.01;
        cost_scaling[749] = 0.01;
        cost_scaling[750] = 0.01;
        cost_scaling[751] = 0.01;
        cost_scaling[752] = 0.01;
        cost_scaling[753] = 0.01;
        cost_scaling[754] = 0.01;
        cost_scaling[755] = 0.01;
        cost_scaling[756] = 0.01;
        cost_scaling[757] = 0.01;
        cost_scaling[758] = 0.01;
        cost_scaling[759] = 0.01;
        cost_scaling[760] = 0.01;
        cost_scaling[761] = 0.01;
        cost_scaling[762] = 0.01;
        cost_scaling[763] = 0.01;
        cost_scaling[764] = 0.01;
        cost_scaling[765] = 0.01;
        cost_scaling[766] = 0.01;
        cost_scaling[767] = 0.01;
        cost_scaling[768] = 0.01;
        cost_scaling[769] = 0.01;
        cost_scaling[770] = 0.01;
        cost_scaling[771] = 0.01;
        cost_scaling[772] = 0.01;
        cost_scaling[773] = 0.01;
        cost_scaling[774] = 0.01;
        cost_scaling[775] = 0.01;
        cost_scaling[776] = 0.01;
        cost_scaling[777] = 0.01;
        cost_scaling[778] = 0.01;
        cost_scaling[779] = 0.01;
        cost_scaling[780] = 0.01;
        cost_scaling[781] = 0.01;
        cost_scaling[782] = 0.01;
        cost_scaling[783] = 0.01;
        cost_scaling[784] = 0.01;
        cost_scaling[785] = 0.01;
        cost_scaling[786] = 0.01;
        cost_scaling[787] = 0.01;
        cost_scaling[788] = 0.01;
        cost_scaling[789] = 0.01;
        cost_scaling[790] = 0.01;
        cost_scaling[791] = 0.01;
        cost_scaling[792] = 0.01;
        cost_scaling[793] = 0.01;
        cost_scaling[794] = 0.01;
        cost_scaling[795] = 0.01;
        cost_scaling[796] = 0.01;
        cost_scaling[797] = 0.01;
        cost_scaling[798] = 0.01;
        cost_scaling[799] = 0.01;
        cost_scaling[800] = 0.01;
        cost_scaling[801] = 0.01;
        cost_scaling[802] = 0.01;
        cost_scaling[803] = 0.01;
        cost_scaling[804] = 0.01;
        cost_scaling[805] = 0.01;
        cost_scaling[806] = 0.01;
        cost_scaling[807] = 0.01;
        cost_scaling[808] = 0.01;
        cost_scaling[809] = 0.01;
        cost_scaling[810] = 0.01;
        cost_scaling[811] = 0.01;
        cost_scaling[812] = 0.01;
        cost_scaling[813] = 0.01;
        cost_scaling[814] = 0.01;
        cost_scaling[815] = 0.01;
        cost_scaling[816] = 0.01;
        cost_scaling[817] = 0.01;
        cost_scaling[818] = 0.01;
        cost_scaling[819] = 0.01;
        cost_scaling[820] = 0.01;
        cost_scaling[821] = 0.01;
        cost_scaling[822] = 0.01;
        cost_scaling[823] = 0.01;
        cost_scaling[824] = 0.01;
        cost_scaling[825] = 0.01;
        cost_scaling[826] = 0.01;
        cost_scaling[827] = 0.01;
        cost_scaling[828] = 0.01;
        cost_scaling[829] = 0.01;
        cost_scaling[830] = 0.01;
        cost_scaling[831] = 0.01;
        cost_scaling[832] = 0.01;
        cost_scaling[833] = 0.01;
        cost_scaling[834] = 0.01;
        cost_scaling[835] = 0.01;
        cost_scaling[836] = 0.01;
        cost_scaling[837] = 0.01;
        cost_scaling[838] = 0.01;
        cost_scaling[839] = 0.01;
        cost_scaling[840] = 0.01;
        cost_scaling[841] = 0.01;
        cost_scaling[842] = 0.01;
        cost_scaling[843] = 0.01;
        cost_scaling[844] = 0.01;
        cost_scaling[845] = 0.01;
        cost_scaling[846] = 0.01;
        cost_scaling[847] = 0.01;
        cost_scaling[848] = 0.01;
        cost_scaling[849] = 0.01;
        cost_scaling[850] = 0.01;
        cost_scaling[851] = 0.01;
        cost_scaling[852] = 0.01;
        cost_scaling[853] = 0.01;
        cost_scaling[854] = 0.01;
        cost_scaling[855] = 0.01;
        cost_scaling[856] = 0.01;
        cost_scaling[857] = 0.01;
        cost_scaling[858] = 0.01;
        cost_scaling[859] = 0.01;
        cost_scaling[860] = 0.01;
        cost_scaling[861] = 0.01;
        cost_scaling[862] = 0.01;
        cost_scaling[863] = 0.01;
        cost_scaling[864] = 0.01;
        cost_scaling[865] = 0.01;
        cost_scaling[866] = 0.01;
        cost_scaling[867] = 0.01;
        cost_scaling[868] = 0.01;
        cost_scaling[869] = 0.01;
        cost_scaling[870] = 0.01;
        cost_scaling[871] = 0.01;
        cost_scaling[872] = 0.01;
        cost_scaling[873] = 0.01;
        cost_scaling[874] = 0.01;
        cost_scaling[875] = 0.01;
        cost_scaling[876] = 0.01;
        cost_scaling[877] = 0.01;
        cost_scaling[878] = 0.01;
        cost_scaling[879] = 0.01;
        cost_scaling[880] = 0.01;
        cost_scaling[881] = 0.01;
        cost_scaling[882] = 0.01;
        cost_scaling[883] = 0.01;
        cost_scaling[884] = 0.01;
        cost_scaling[885] = 0.01;
        cost_scaling[886] = 0.01;
        cost_scaling[887] = 0.01;
        cost_scaling[888] = 0.01;
        cost_scaling[889] = 0.01;
        cost_scaling[890] = 0.01;
        cost_scaling[891] = 0.01;
        cost_scaling[892] = 0.01;
        cost_scaling[893] = 0.01;
        cost_scaling[894] = 0.01;
        cost_scaling[895] = 0.01;
        cost_scaling[896] = 0.01;
        cost_scaling[897] = 0.01;
        cost_scaling[898] = 0.01;
        cost_scaling[899] = 0.01;
        cost_scaling[900] = 0.01;
        cost_scaling[901] = 0.01;
        cost_scaling[902] = 0.01;
        cost_scaling[903] = 0.01;
        cost_scaling[904] = 0.01;
        cost_scaling[905] = 0.01;
        cost_scaling[906] = 0.01;
        cost_scaling[907] = 0.01;
        cost_scaling[908] = 0.01;
        cost_scaling[909] = 0.01;
        cost_scaling[910] = 0.01;
        cost_scaling[911] = 0.01;
        cost_scaling[912] = 0.01;
        cost_scaling[913] = 0.01;
        cost_scaling[914] = 0.01;
        cost_scaling[915] = 0.01;
        cost_scaling[916] = 0.01;
        cost_scaling[917] = 0.01;
        cost_scaling[918] = 0.01;
        cost_scaling[919] = 0.01;
        cost_scaling[920] = 0.01;
        cost_scaling[921] = 0.01;
        cost_scaling[922] = 0.01;
        cost_scaling[923] = 0.01;
        cost_scaling[924] = 0.01;
        cost_scaling[925] = 0.01;
        cost_scaling[926] = 0.01;
        cost_scaling[927] = 0.01;
        cost_scaling[928] = 0.01;
        cost_scaling[929] = 0.01;
        cost_scaling[930] = 0.01;
        cost_scaling[931] = 0.01;
        cost_scaling[932] = 0.01;
        cost_scaling[933] = 0.01;
        cost_scaling[934] = 0.01;
        cost_scaling[935] = 0.01;
        cost_scaling[936] = 0.01;
        cost_scaling[937] = 0.01;
        cost_scaling[938] = 0.01;
        cost_scaling[939] = 0.01;
        cost_scaling[940] = 0.01;
        cost_scaling[941] = 0.01;
        cost_scaling[942] = 0.01;
        cost_scaling[943] = 0.01;
        cost_scaling[944] = 0.01;
        cost_scaling[945] = 0.01;
        cost_scaling[946] = 0.01;
        cost_scaling[947] = 0.01;
        cost_scaling[948] = 0.01;
        cost_scaling[949] = 0.01;
        cost_scaling[950] = 0.01;
        cost_scaling[951] = 0.01;
        cost_scaling[952] = 0.01;
        cost_scaling[953] = 0.01;
        cost_scaling[954] = 0.01;
        cost_scaling[955] = 0.01;
        cost_scaling[956] = 0.01;
        cost_scaling[957] = 0.01;
        cost_scaling[958] = 0.01;
        cost_scaling[959] = 0.01;
        cost_scaling[960] = 0.01;
        cost_scaling[961] = 0.01;
        cost_scaling[962] = 0.01;
        cost_scaling[963] = 0.01;
        cost_scaling[964] = 0.01;
        cost_scaling[965] = 0.01;
        cost_scaling[966] = 0.01;
        cost_scaling[967] = 0.01;
        cost_scaling[968] = 0.01;
        cost_scaling[969] = 0.01;
        cost_scaling[970] = 0.01;
        cost_scaling[971] = 0.01;
        cost_scaling[972] = 0.01;
        cost_scaling[973] = 0.01;
        cost_scaling[974] = 0.01;
        cost_scaling[975] = 0.01;
        cost_scaling[976] = 0.01;
        cost_scaling[977] = 0.01;
        cost_scaling[978] = 0.01;
        cost_scaling[979] = 0.01;
        cost_scaling[980] = 0.01;
        cost_scaling[981] = 0.01;
        cost_scaling[982] = 0.01;
        cost_scaling[983] = 0.01;
        cost_scaling[984] = 0.01;
        cost_scaling[985] = 0.01;
        cost_scaling[986] = 0.01;
        cost_scaling[987] = 0.01;
        cost_scaling[988] = 0.01;
        cost_scaling[989] = 0.01;
        cost_scaling[990] = 0.01;
        cost_scaling[991] = 0.01;
        cost_scaling[992] = 0.01;
        cost_scaling[993] = 0.01;
        cost_scaling[994] = 0.01;
        cost_scaling[995] = 0.01;
        cost_scaling[996] = 0.01;
        cost_scaling[997] = 0.01;
        cost_scaling[998] = 0.01;
        cost_scaling[999] = 0.01;
        cost_scaling[1000] = 1;
        for (int i = 0; i <= N; i++)
        {
            ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "scaling", &cost_scaling[i]);
        }
        free(cost_scaling);
    }



    /**** Dynamics ****/
    for (int i = 0; i < N; i++)
    {
        ocp_nlp_dynamics_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "expl_vde_forw", &capsule->expl_vde_forw[i]);
        ocp_nlp_dynamics_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "expl_ode_fun", &capsule->expl_ode_fun[i]);
        ocp_nlp_dynamics_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "expl_vde_adj", &capsule->expl_vde_adj[i]);
    }

    /**** Cost ****/
    double* yref_0 = calloc(NY0, sizeof(double));
    // change only the non-zero elements:
    yref_0[0] = 2;
    yref_0[6] = 1;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "yref", yref_0);
    free(yref_0);

   double* W_0 = calloc(NY0*NY0, sizeof(double));
    // change only the non-zero elements:
    W_0[0+(NY0) * 0] = 0.1;
    W_0[1+(NY0) * 1] = 0.25330295910584444;
    W_0[2+(NY0) * 2] = 0.25330295910584444;
    W_0[3+(NY0) * 3] = 0.08;
    W_0[4+(NY0) * 4] = 0.08;
    W_0[5+(NY0) * 5] = 0.08;
    W_0[6+(NY0) * 6] = 1;
    W_0[7+(NY0) * 7] = 5;
    W_0[8+(NY0) * 8] = 5;
    W_0[9+(NY0) * 9] = 5;
    W_0[10+(NY0) * 10] = 2.7356719583431204;
    W_0[11+(NY0) * 11] = 2.7356719583431204;
    W_0[12+(NY0) * 12] = 3.64756261112416;
    W_0[13+(NY0) * 13] = 0.027777777777777776;
    W_0[14+(NY0) * 14] = 0.027777777777777776;
    W_0[15+(NY0) * 15] = 0.027777777777777776;
    W_0[16+(NY0) * 16] = 0.000025;
    W_0[17+(NY0) * 17] = 0.000025;
    W_0[18+(NY0) * 18] = 0.000025;
    W_0[19+(NY0) * 19] = 0.0005;
    W_0[20+(NY0) * 20] = 0.0005;
    W_0[21+(NY0) * 21] = 0.0005;
    W_0[22+(NY0) * 22] = 0.000005;
    W_0[23+(NY0) * 23] = 0.000005;
    W_0[24+(NY0) * 24] = 0.000005;
    W_0[25+(NY0) * 25] = 0.00000625;
    W_0[26+(NY0) * 26] = 1.1111111111111112;
    W_0[27+(NY0) * 27] = 1.1111111111111112;
    W_0[28+(NY0) * 28] = 1.1111111111111112;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "W", W_0);
    free(W_0);
    double* yref = calloc(NY, sizeof(double));
    // change only the non-zero elements:
    yref[0] = 2;
    yref[6] = 1;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "yref", yref);
    }
    free(yref);
    double* W = calloc(NY*NY, sizeof(double));
    // change only the non-zero elements:
    W[0+(NY) * 0] = 0.1;
    W[1+(NY) * 1] = 0.25330295910584444;
    W[2+(NY) * 2] = 0.25330295910584444;
    W[3+(NY) * 3] = 0.08;
    W[4+(NY) * 4] = 0.08;
    W[5+(NY) * 5] = 0.08;
    W[6+(NY) * 6] = 1;
    W[7+(NY) * 7] = 5;
    W[8+(NY) * 8] = 5;
    W[9+(NY) * 9] = 5;
    W[10+(NY) * 10] = 2.7356719583431204;
    W[11+(NY) * 11] = 2.7356719583431204;
    W[12+(NY) * 12] = 3.64756261112416;
    W[13+(NY) * 13] = 0.027777777777777776;
    W[14+(NY) * 14] = 0.027777777777777776;
    W[15+(NY) * 15] = 0.027777777777777776;
    W[16+(NY) * 16] = 0.000025;
    W[17+(NY) * 17] = 0.000025;
    W[18+(NY) * 18] = 0.000025;
    W[19+(NY) * 19] = 0.0005;
    W[20+(NY) * 20] = 0.0005;
    W[21+(NY) * 21] = 0.0005;
    W[22+(NY) * 22] = 0.000005;
    W[23+(NY) * 23] = 0.000005;
    W[24+(NY) * 24] = 0.000005;
    W[25+(NY) * 25] = 0.00000625;
    W[26+(NY) * 26] = 1.1111111111111112;
    W[27+(NY) * 27] = 1.1111111111111112;
    W[28+(NY) * 28] = 1.1111111111111112;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "W", W);
    }
    free(W);
    double* yref_e = calloc(NYN, sizeof(double));
    // change only the non-zero elements:
    yref_e[0] = 2;
    yref_e[6] = 1;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "yref", yref_e);
    free(yref_e);

    double* W_e = calloc(NYN*NYN, sizeof(double));
    // change only the non-zero elements:
    W_e[0+(NYN) * 0] = 1;
    W_e[1+(NYN) * 1] = 2.5330295910584444;
    W_e[2+(NYN) * 2] = 2.5330295910584444;
    W_e[3+(NYN) * 3] = 0.8;
    W_e[4+(NYN) * 4] = 0.8;
    W_e[5+(NYN) * 5] = 0.8;
    W_e[6+(NYN) * 6] = 10;
    W_e[7+(NYN) * 7] = 50;
    W_e[8+(NYN) * 8] = 50;
    W_e[9+(NYN) * 9] = 50;
    W_e[10+(NYN) * 10] = 27.356719583431204;
    W_e[11+(NYN) * 11] = 27.356719583431204;
    W_e[12+(NYN) * 12] = 36.4756261112416;
    W_e[13+(NYN) * 13] = 0.2777777777777778;
    W_e[14+(NYN) * 14] = 0.2777777777777778;
    W_e[15+(NYN) * 15] = 0.2777777777777778;
    W_e[16+(NYN) * 16] = 0.00025;
    W_e[17+(NYN) * 17] = 0.00025;
    W_e[18+(NYN) * 18] = 0.00025;
    W_e[19+(NYN) * 19] = 0.005;
    W_e[20+(NYN) * 20] = 0.005;
    W_e[21+(NYN) * 21] = 0.005;
    W_e[22+(NYN) * 22] = 0.00005;
    W_e[23+(NYN) * 23] = 0.00005;
    W_e[24+(NYN) * 24] = 0.00005;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "W", W_e);
    free(W_e);
    ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, 0, "nls_y_fun", &capsule->cost_y_0_fun);
    ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, 0, "nls_y_fun_jac", &capsule->cost_y_0_fun_jac_ut_xt);
    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "nls_y_fun", &capsule->cost_y_fun[i-1]);
        ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "nls_y_fun_jac", &capsule->cost_y_fun_jac_ut_xt[i-1]);
    }
    ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, N, "nls_y_fun", &capsule->cost_y_e_fun);
    ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, N, "nls_y_fun_jac", &capsule->cost_y_e_fun_jac_ut_xt);







    /**** Constraints ****/

    // bounds for initial stage
    // x0
    int* idxbx0 = malloc(NBX0 * sizeof(int));
    idxbx0[0] = 0;
    idxbx0[1] = 1;
    idxbx0[2] = 2;
    idxbx0[3] = 3;
    idxbx0[4] = 4;
    idxbx0[5] = 5;
    idxbx0[6] = 6;
    idxbx0[7] = 7;
    idxbx0[8] = 8;
    idxbx0[9] = 9;
    idxbx0[10] = 10;
    idxbx0[11] = 11;
    idxbx0[12] = 12;

    double* lubx0 = calloc(2*NBX0, sizeof(double));
    double* lbx0 = lubx0;
    double* ubx0 = lubx0 + NBX0;
    // change only the non-zero elements:
    lbx0[6] = 1;
    ubx0[6] = 1;

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "idxbx", idxbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "lbx", lbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "ubx", ubx0);
    free(idxbx0);
    free(lubx0);
    // idxbxe_0
    int* idxbxe_0 = malloc(13 * sizeof(int));
    idxbxe_0[0] = 0;
    idxbxe_0[1] = 1;
    idxbxe_0[2] = 2;
    idxbxe_0[3] = 3;
    idxbxe_0[4] = 4;
    idxbxe_0[5] = 5;
    idxbxe_0[6] = 6;
    idxbxe_0[7] = 7;
    idxbxe_0[8] = 8;
    idxbxe_0[9] = 9;
    idxbxe_0[10] = 10;
    idxbxe_0[11] = 11;
    idxbxe_0[12] = 12;
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "idxbxe", idxbxe_0);
    free(idxbxe_0);












    /* constraints that are the same for initial and intermediate */
    // u
    int* idxbu = malloc(NBU * sizeof(int));
    idxbu[0] = 0;
    idxbu[1] = 1;
    idxbu[2] = 2;
    idxbu[3] = 3;
    double* lubu = calloc(2*NBU, sizeof(double));
    double* lbu = lubu;
    double* ubu = lubu + NBU;
    lbu[0] = -40;
    ubu[0] = 40;
    lbu[1] = -3;
    ubu[1] = 3;
    lbu[2] = -3;
    ubu[2] = 3;
    lbu[3] = -3;
    ubu[3] = 3;

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "idxbu", idxbu);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "lbu", lbu);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "ubu", ubu);
    }
    free(idxbu);
    free(lubu);






    /* Path constraints */

    // x
    int* idxbx = malloc(NBX * sizeof(int));
    idxbx[0] = 2;
    idxbx[1] = 3;
    idxbx[2] = 4;
    idxbx[3] = 5;
    idxbx[4] = -3;
    idxbx[5] = -2;
    idxbx[6] = -1;
    double* lubx = calloc(2*NBX, sizeof(double));
    double* lbx = lubx;
    double* ubx = lubx + NBX;
    ubx[0] = 100;
    lbx[1] = -5;
    ubx[1] = 5;
    lbx[2] = -5;
    ubx[2] = 5;
    lbx[3] = -5;
    ubx[3] = 5;
    lbx[4] = -1.0471975511965976;
    ubx[4] = 1.0471975511965976;
    lbx[5] = -1.0471975511965976;
    ubx[5] = 1.0471975511965976;
    lbx[6] = -1.0471975511965976;
    ubx[6] = 1.0471975511965976;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "idxbx", idxbx);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "lbx", lbx);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "ubx", ubx);
    }
    free(idxbx);
    free(lubx);













    /* terminal constraints */




















}


static void quadrotor_ode_acados_create_set_opts(quadrotor_ode_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    void *nlp_opts = capsule->nlp_opts;

    /************************************************
    *  opts
    ************************************************/



    int fixed_hess = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "fixed_hess", &fixed_hess);
    double globalization_alpha_min = 0.05;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "globalization_alpha_min", &globalization_alpha_min);

    double globalization_alpha_reduction = 0.7;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "globalization_alpha_reduction", &globalization_alpha_reduction);



    int globalization_line_search_use_sufficient_descent = 0;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "globalization_line_search_use_sufficient_descent", &globalization_line_search_use_sufficient_descent);

    int globalization_use_SOC = 0;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "globalization_use_SOC", &globalization_use_SOC);

    double globalization_eps_sufficient_descent = 0.0001;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "globalization_eps_sufficient_descent", &globalization_eps_sufficient_descent);

    int with_solution_sens_wrt_params = false;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "with_solution_sens_wrt_params", &with_solution_sens_wrt_params);

    int with_value_sens_wrt_params = false;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "with_value_sens_wrt_params", &with_value_sens_wrt_params);

    double solution_sens_qp_t_lam_min = 0.000000001;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "solution_sens_qp_t_lam_min", &solution_sens_qp_t_lam_min);

    int globalization_full_step_dual = 0;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "globalization_full_step_dual", &globalization_full_step_dual);

    // set collocation type (relevant for implicit integrators)
    sim_collocation_type collocation_type = GAUSS_LEGENDRE;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_collocation_type", &collocation_type);

    // set up sim_method_num_steps
    // all sim_method_num_steps are identical
    int sim_method_num_steps = 1;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_num_steps", &sim_method_num_steps);

    // set up sim_method_num_stages
    // all sim_method_num_stages are identical
    int sim_method_num_stages = 4;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_num_stages", &sim_method_num_stages);

    int newton_iter_val = 3;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_newton_iter", &newton_iter_val);

    double newton_tol_val = 0;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_newton_tol", &newton_tol_val);

    // set up sim_method_jac_reuse
    bool tmp_bool = (bool) 0;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_jac_reuse", &tmp_bool);

    double levenberg_marquardt = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "levenberg_marquardt", &levenberg_marquardt);

    /* options QP solver */
    int qp_solver_cond_N;const int qp_solver_cond_N_ori = 1000;
    qp_solver_cond_N = N < qp_solver_cond_N_ori ? N : qp_solver_cond_N_ori; // use the minimum value here
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_cond_N", &qp_solver_cond_N);

    int nlp_solver_ext_qp_res = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "ext_qp_res", &nlp_solver_ext_qp_res);

    bool store_iterates = false;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "store_iterates", &store_iterates);
    int log_primal_step_norm = false;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "log_primal_step_norm", &log_primal_step_norm);

    int log_dual_step_norm = false;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "log_dual_step_norm", &log_dual_step_norm);

    double nlp_solver_tol_min_step_norm = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "tol_min_step_norm", &nlp_solver_tol_min_step_norm);
    // set HPIPM mode: should be done before setting other QP solver options
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_hpipm_mode", "BALANCE");



    int qp_solver_t0_init = 2;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_t0_init", &qp_solver_t0_init);




    // set SQP specific options
    double nlp_solver_tol_stat = 0.000001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "tol_stat", &nlp_solver_tol_stat);

    double nlp_solver_tol_eq = 0.000001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "tol_eq", &nlp_solver_tol_eq);

    double nlp_solver_tol_ineq = 0.000001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "tol_ineq", &nlp_solver_tol_ineq);

    double nlp_solver_tol_comp = 0.000001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "tol_comp", &nlp_solver_tol_comp);

    int nlp_solver_max_iter = 200;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "max_iter", &nlp_solver_max_iter);

    // set options for adaptive Levenberg-Marquardt Update
    bool with_adaptive_levenberg_marquardt = false;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "with_adaptive_levenberg_marquardt", &with_adaptive_levenberg_marquardt);

    double adaptive_levenberg_marquardt_lam = 5;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "adaptive_levenberg_marquardt_lam", &adaptive_levenberg_marquardt_lam);

    double adaptive_levenberg_marquardt_mu_min = 0.0000000000000001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "adaptive_levenberg_marquardt_mu_min", &adaptive_levenberg_marquardt_mu_min);

    double adaptive_levenberg_marquardt_mu0 = 0.001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "adaptive_levenberg_marquardt_mu0", &adaptive_levenberg_marquardt_mu0);

    double adaptive_levenberg_marquardt_obj_scalar = 2;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "adaptive_levenberg_marquardt_obj_scalar", &adaptive_levenberg_marquardt_obj_scalar);

    bool eval_residual_at_max_iter = false;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "eval_residual_at_max_iter", &eval_residual_at_max_iter);

    // QP scaling
    double qpscaling_ub_max_abs_eig = 100000;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qpscaling_ub_max_abs_eig", &qpscaling_ub_max_abs_eig);

    double qpscaling_lb_norm_inf_grad_obj = 0.0001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qpscaling_lb_norm_inf_grad_obj", &qpscaling_lb_norm_inf_grad_obj);

    qpscaling_scale_objective_type qpscaling_scale_objective = NO_OBJECTIVE_SCALING;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qpscaling_scale_objective", &qpscaling_scale_objective);

    ocp_nlp_qpscaling_constraint_type qpscaling_scale_constraints = NO_CONSTRAINT_SCALING;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qpscaling_scale_constraints", &qpscaling_scale_constraints);

    // NLP QP tol strategy
    ocp_nlp_qp_tol_strategy_t nlp_qp_tol_strategy = FIXED_QP_TOL;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "nlp_qp_tol_strategy", &nlp_qp_tol_strategy);

    double nlp_qp_tol_reduction_factor = 0.1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "nlp_qp_tol_reduction_factor", &nlp_qp_tol_reduction_factor);

    double nlp_qp_tol_safety_factor = 0.1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "nlp_qp_tol_safety_factor", &nlp_qp_tol_safety_factor);

    double nlp_qp_tol_min_stat = 0.000000001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "nlp_qp_tol_min_stat", &nlp_qp_tol_min_stat);

    double nlp_qp_tol_min_eq = 0.0000000001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "nlp_qp_tol_min_eq", &nlp_qp_tol_min_eq);

    double nlp_qp_tol_min_ineq = 0.0000000001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "nlp_qp_tol_min_ineq", &nlp_qp_tol_min_ineq);

    double nlp_qp_tol_min_comp = 0.00000000001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "nlp_qp_tol_min_comp", &nlp_qp_tol_min_comp);

    bool with_anderson_acceleration = false;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "with_anderson_acceleration", &with_anderson_acceleration);

    int qp_solver_iter_max = 50;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_iter_max", &qp_solver_iter_max);



    int print_level = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "print_level", &print_level);
    int qp_solver_cond_ric_alg = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_cond_ric_alg", &qp_solver_cond_ric_alg);

    int qp_solver_ric_alg = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_ric_alg", &qp_solver_ric_alg);


    int ext_cost_num_hess = 0;
}


/**
 * Internal function for quadrotor_ode_acados_create: step 7
 */
void quadrotor_ode_acados_set_nlp_out(quadrotor_ode_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;
    ocp_nlp_out* nlp_out = capsule->nlp_out;
    ocp_nlp_in* nlp_in = capsule->nlp_in;

    // initialize primal solution
    double* xu0 = calloc(NX+NU, sizeof(double));
    double* x0 = xu0;

    // initialize with x0
    x0[6] = 1;


    double* u0 = xu0 + NX;

    for (int i = 0; i < N; i++)
    {
        // x0
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "x", x0);
        // u0
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "u", u0);
    }
    ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, N, "x", x0);
    free(xu0);
}


/**
 * Internal function for quadrotor_ode_acados_create: step 9
 */
int quadrotor_ode_acados_create_precompute(quadrotor_ode_solver_capsule* capsule) {
    int status = ocp_nlp_precompute(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    if (status != ACADOS_SUCCESS) {
        printf("\nocp_nlp_precompute failed!\n\n");
        exit(1);
    }

    return status;
}


int quadrotor_ode_acados_create_with_discretization(quadrotor_ode_solver_capsule* capsule, int N, double* new_time_steps)
{
    // If N does not match the number of shooting intervals used for code generation, new_time_steps must be given.
    if (N != QUADROTOR_ODE_N && !new_time_steps) {
        fprintf(stderr, "quadrotor_ode_acados_create_with_discretization: new_time_steps is NULL " \
            "but the number of shooting intervals (= %d) differs from the number of " \
            "shooting intervals (= %d) during code generation! Please provide a new vector of time_stamps!\n", \
             N, QUADROTOR_ODE_N);
        return 1;
    }

    // number of expected runtime parameters
    capsule->nlp_np = NP;

    // 1) create and set nlp_solver_plan; create nlp_config
    capsule->nlp_solver_plan = ocp_nlp_plan_create(N);
    quadrotor_ode_acados_create_set_plan(capsule->nlp_solver_plan, N);
    capsule->nlp_config = ocp_nlp_config_create(*capsule->nlp_solver_plan);

    // 2) create and set dimensions
    capsule->nlp_dims = quadrotor_ode_acados_create_setup_dimensions(capsule);

    // 3) create and set nlp_opts
    capsule->nlp_opts = ocp_nlp_solver_opts_create(capsule->nlp_config, capsule->nlp_dims);
    quadrotor_ode_acados_create_set_opts(capsule);

    // 4) create and set nlp_out
    // 4.1) nlp_out
    capsule->nlp_out = ocp_nlp_out_create(capsule->nlp_config, capsule->nlp_dims);
    // 4.2) sens_out
    capsule->sens_out = ocp_nlp_out_create(capsule->nlp_config, capsule->nlp_dims);
    quadrotor_ode_acados_set_nlp_out(capsule);

    // 5) create nlp_in
    capsule->nlp_in = ocp_nlp_in_create(capsule->nlp_config, capsule->nlp_dims);

    // 6) setup functions, nlp_in and default parameters
    quadrotor_ode_acados_create_setup_functions(capsule);
    quadrotor_ode_acados_setup_nlp_in(capsule, N, new_time_steps);
    quadrotor_ode_acados_create_set_default_parameters(capsule);

    // 7) create solver
    capsule->nlp_solver = ocp_nlp_solver_create(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_opts, capsule->nlp_in);


    // 8) do precomputations
    int status = quadrotor_ode_acados_create_precompute(capsule);

    return status;
}

/**
 * This function is for updating an already initialized solver with a different number of qp_cond_N. It is useful for code reuse after code export.
 */
int quadrotor_ode_acados_update_qp_solver_cond_N(quadrotor_ode_solver_capsule* capsule, int qp_solver_cond_N)
{
    // 1) destroy solver
    ocp_nlp_solver_destroy(capsule->nlp_solver);

    // 2) set new value for "qp_cond_N"
    const int N = capsule->nlp_solver_plan->N;
    if(qp_solver_cond_N > N)
        printf("Warning: qp_solver_cond_N = %d > N = %d\n", qp_solver_cond_N, N);
    ocp_nlp_solver_opts_set(capsule->nlp_config, capsule->nlp_opts, "qp_cond_N", &qp_solver_cond_N);

    // 3) continue with the remaining steps from quadrotor_ode_acados_create_with_discretization(...):
    // -> 8) create solver
    capsule->nlp_solver = ocp_nlp_solver_create(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_opts, capsule->nlp_in);

    // -> 9) do precomputations
    int status = quadrotor_ode_acados_create_precompute(capsule);
    return status;
}


int quadrotor_ode_acados_reset(quadrotor_ode_solver_capsule* capsule, int reset_qp_solver_mem)
{

    // set initialization to all zeros

    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;
    ocp_nlp_out* nlp_out = capsule->nlp_out;
    ocp_nlp_in* nlp_in = capsule->nlp_in;
    ocp_nlp_solver* nlp_solver = capsule->nlp_solver;

    double* buffer = calloc(NX+NU+NZ+2*NS+2*NSN+2*NS0+NBX+NBU+NG+NH+NPHI+NBX0+NBXN+NHN+NH0+NPHIN+NGN, sizeof(double));

    for(int i=0; i<N+1; i++)
    {
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "x", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "u", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "sl", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "su", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "lam", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "z", buffer);
        if (i<N)
        {
            ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "pi", buffer);
        }
    }
    // get qp_status: if NaN -> reset memory
    int qp_status;
    ocp_nlp_get(capsule->nlp_solver, "qp_status", &qp_status);
    if (reset_qp_solver_mem || (qp_status == 3))
    {
        // printf("\nin reset qp_status %d -> resetting QP memory\n", qp_status);
        ocp_nlp_solver_reset_qp_memory(nlp_solver, nlp_in, nlp_out);
    }

    free(buffer);
    return 0;
}




int quadrotor_ode_acados_update_params(quadrotor_ode_solver_capsule* capsule, int stage, double *p, int np)
{
    int solver_status = 0;

    int casadi_np = 9;
    if (casadi_np != np) {
        printf("acados_update_params: trying to set %i parameters for external functions."
            " External function has %i parameters. Exiting.\n", np, casadi_np);
        exit(1);
    }
    ocp_nlp_in_set(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, stage, "parameter_values", p);

    return solver_status;
}


int quadrotor_ode_acados_update_params_sparse(quadrotor_ode_solver_capsule * capsule, int stage, int *idx, double *p, int n_update)
{
    ocp_nlp_in_set_params_sparse(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, stage, idx, p, n_update);

    return 0;
}


int quadrotor_ode_acados_set_p_global_and_precompute_dependencies(quadrotor_ode_solver_capsule* capsule, double* data, int data_len)
{

    // printf("No global_data, quadrotor_ode_acados_set_p_global_and_precompute_dependencies does nothing.\n");
    return 0;
}




int quadrotor_ode_acados_solve(quadrotor_ode_solver_capsule* capsule)
{
    // solve NLP
    int solver_status = ocp_nlp_solve(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    return solver_status;
}



int quadrotor_ode_acados_setup_qp_matrices_and_factorize(quadrotor_ode_solver_capsule* capsule)
{
    int solver_status = ocp_nlp_setup_qp_matrices_and_factorize(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    return solver_status;
}






int quadrotor_ode_acados_free(quadrotor_ode_solver_capsule* capsule)
{
    // before destroying, keep some info
    const int N = capsule->nlp_solver_plan->N;
    // free memory
    ocp_nlp_solver_opts_destroy(capsule->nlp_opts);
    ocp_nlp_in_destroy(capsule->nlp_in);
    ocp_nlp_out_destroy(capsule->nlp_out);
    ocp_nlp_out_destroy(capsule->sens_out);
    ocp_nlp_solver_destroy(capsule->nlp_solver);
    ocp_nlp_dims_destroy(capsule->nlp_dims);
    ocp_nlp_config_destroy(capsule->nlp_config);
    ocp_nlp_plan_destroy(capsule->nlp_solver_plan);

    /* free external function */
    // dynamics
    for (int i = 0; i < N; i++)
    {
        external_function_external_param_casadi_free(&capsule->expl_vde_forw[i]);
        external_function_external_param_casadi_free(&capsule->expl_ode_fun[i]);
        external_function_external_param_casadi_free(&capsule->expl_vde_adj[i]);
    }
    free(capsule->expl_vde_adj);
    free(capsule->expl_vde_forw);
    free(capsule->expl_ode_fun);

    // cost
    external_function_external_param_casadi_free(&capsule->cost_y_0_fun);
    external_function_external_param_casadi_free(&capsule->cost_y_0_fun_jac_ut_xt);
    for (int i = 0; i < N - 1; i++)
    {
        external_function_external_param_casadi_free(&capsule->cost_y_fun[i]);
        external_function_external_param_casadi_free(&capsule->cost_y_fun_jac_ut_xt[i]);
    }
    free(capsule->cost_y_fun);
    free(capsule->cost_y_fun_jac_ut_xt);
    external_function_external_param_casadi_free(&capsule->cost_y_e_fun);
    external_function_external_param_casadi_free(&capsule->cost_y_e_fun_jac_ut_xt);

    // constraints



    return 0;
}


void quadrotor_ode_acados_print_stats(quadrotor_ode_solver_capsule* capsule)
{
    int nlp_iter, stat_m, stat_n, tmp_int;
    ocp_nlp_get(capsule->nlp_solver, "nlp_iter", &nlp_iter);
    ocp_nlp_get(capsule->nlp_solver, "stat_n", &stat_n);
    ocp_nlp_get(capsule->nlp_solver, "stat_m", &stat_m);


    int stat_n_max = 16;
    if (stat_n > stat_n_max)
    {
        printf("stat_n_max = %d is too small, increase it in the template!\n", stat_n_max);
        exit(1);
    }
    double stat[3200];
    ocp_nlp_get(capsule->nlp_solver, "statistics", stat);

    int nrow = nlp_iter+1 < stat_m ? nlp_iter+1 : stat_m;


    printf("iter\tres_stat\tres_eq\t\tres_ineq\tres_comp\tqp_stat\tqp_iter\talpha");
    if (stat_n > 8)
        printf("\t\tqp_res_stat\tqp_res_eq\tqp_res_ineq\tqp_res_comp");
    printf("\n");
    for (int i = 0; i < nrow; i++)
    {
        for (int j = 0; j < stat_n + 1; j++)
        {
            if (j == 0 || j == 5 || j == 6)
            {
                tmp_int = (int) stat[i + j * nrow];
                printf("%d\t", tmp_int);
            }
            else
            {
                printf("%e\t", stat[i + j * nrow]);
            }
        }
        printf("\n");
    }
}

int quadrotor_ode_acados_custom_update(quadrotor_ode_solver_capsule* capsule, double* data, int data_len)
{
    (void)capsule;
    (void)data;
    (void)data_len;
    printf("\ndummy function that can be called in between solver calls to update parameters or numerical data efficiently in C.\n");
    printf("nothing set yet..\n");
    return 1;

}



ocp_nlp_in *quadrotor_ode_acados_get_nlp_in(quadrotor_ode_solver_capsule* capsule) { return capsule->nlp_in; }
ocp_nlp_out *quadrotor_ode_acados_get_nlp_out(quadrotor_ode_solver_capsule* capsule) { return capsule->nlp_out; }
ocp_nlp_out *quadrotor_ode_acados_get_sens_out(quadrotor_ode_solver_capsule* capsule) { return capsule->sens_out; }
ocp_nlp_solver *quadrotor_ode_acados_get_nlp_solver(quadrotor_ode_solver_capsule* capsule) { return capsule->nlp_solver; }
ocp_nlp_config *quadrotor_ode_acados_get_nlp_config(quadrotor_ode_solver_capsule* capsule) { return capsule->nlp_config; }
void *quadrotor_ode_acados_get_nlp_opts(quadrotor_ode_solver_capsule* capsule) { return capsule->nlp_opts; }
ocp_nlp_dims *quadrotor_ode_acados_get_nlp_dims(quadrotor_ode_solver_capsule* capsule) { return capsule->nlp_dims; }
ocp_nlp_plan_t *quadrotor_ode_acados_get_nlp_plan(quadrotor_ode_solver_capsule* capsule) { return capsule->nlp_solver_plan; }
