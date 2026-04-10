#include "udf.h"
#include "dynamesh_tools.h"

/* ========================================================= */
/* --- LITERATURE-BASED CONSTANTS (Phase A Laminar VIV) ---  */
/* ========================================================= */
#define D_CYL          1.0      /* Cylinder diameter (m) */
#define U_INF          1.0      /* Free-stream velocity (m/s) */
#define RHO_FLUID      1.0      /* Model fluid density (kg/m^3) */

#define M_STAR         2.0      /* Mass ratio */
#define ZETA           0.007    /* Damping ratio (0.7%) */

/* ========================================================= */
/* --- YOUR SWEEP PARAMETER ---                              */
/* Change U_REDUCED for each simulation run to map lock-in   */
/* Lock-in for these settings usually occurs between Ur=4 to 7 */
/* ========================================================= */
#define U_REDUCED      5.0      

/* Non-dimensional time to release cylinder (let wake form first) */
#define T_STAR_RELEASE 60.0     

static real y_eq = 0.0;
static int is_initialized = 0;

DEFINE_SDOF_PROPERTIES(cylinder_heave, prop, dt, time, dtime)
{
    /* 1. CALCULATE PHYSICAL MASS (m = m* * rho * pi * D^2 / 4) */
    real mass = M_STAR * RHO_FLUID * (M_PI * D_CYL * D_CYL / 4.0);
    
    /* 2. CALCULATE NATURAL FREQUENCY FROM REDUCED VELOCITY */
    /* Ur = U / (fn * D)  =>  fn = U / (Ur * D) */
    real f_n = U_INF / (U_REDUCED * D_CYL);
    real omega_n = 2.0 * M_PI * f_n;
    
    /* 3. CALCULATE SPRING AND DAMPER CONSTANTS */
    real k_stiff = mass * omega_n * omega_n;
    real c_damp  = 2.0 * mass * omega_n * ZETA;
    
    /* 4. CALCULATE RELEASE TIME IN SECONDS */
    real release_time = (T_STAR_RELEASE * D_CYL) / U_INF;

    /* 5. GET CURRENT KINEMATICS */
    real y_curr = DT_CG(dt)[1];
    real v_curr = DT_VEL_CG(dt)[1];

    /* 6. INITIALIZATION & CONSOLE MESSAGE */
    if (!is_initialized)
    {
        y_eq = y_curr;
        Message("\n[VIV UDF] Re=200 Cylinder Initialized.\n");
        Message("  -> Target Ur  = %.2f\n", U_REDUCED);
        Message("  -> Auto-Calc: fn = %.4f Hz, Mass = %.4f kg/m, Stiffness = %.4f N/m\n", f_n, mass, k_stiff);
        Message("  -> Cylinder locked until t = %.2f s (t* = 60)\n\n", release_time);
        is_initialized = 1;
    }

    /* 7. APPLY MASS & INERTIA */
    prop[SDOF_MASS] = mass;
    prop[SDOF_IXX]  = 1.0; 
    prop[SDOF_IYY]  = 1.0;
    prop[SDOF_IZZ]  = 1.0;

    /* 8. CONSTRAIN ALL DOFS EXCEPT Y-TRANSLATION (HEAVE) */
    prop[SDOF_ZERO_TRANS_X] = TRUE;
    prop[SDOF_ZERO_TRANS_Z] = TRUE;
    prop[SDOF_ZERO_ROT_X]   = TRUE;
    prop[SDOF_ZERO_ROT_Y]   = TRUE;
    prop[SDOF_ZERO_ROT_Z]   = TRUE;

    /* 9. HOLD AND RELEASE LOGIC */
    if (time < release_time)
    {
        /* HOLD: Lock Y motion */
        prop[SDOF_ZERO_TRANS_Y] = TRUE;
        prop[SDOF_LOAD_F_Y] = 0.0;
    }
    else
    {
        /* RELEASE: Unlock Y motion and apply spring-damper forces */
        prop[SDOF_ZERO_TRANS_Y] = FALSE;

        real force_spring = -k_stiff * (y_curr - y_eq);
        real force_damp   = -c_damp  * v_curr;
        
        prop[SDOF_LOAD_F_Y] = force_spring + force_damp;
    }
}