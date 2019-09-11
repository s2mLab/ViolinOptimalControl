#ifndef BIORBD_ACADO_INTERFACE_H
#define BIORBD_ACADO_INTERFACE_H

#include <math.h>

#include "BiorbdModel.h"
extern biorbd::Model m;

#ifdef __cplusplus
extern "C" {
#endif

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int long long int
#endif

// Preparing stuff
void fillSparsity();

// Functions
BIORBD_API const char* libforward_dynamics_casadi_name(void);
BIORBD_API int libforward_dynamics_casadi(const casadi_real** arg,
                                                   casadi_real** res,
                                                   casadi_int* iw,
                                                   casadi_real* w,
                                                   void* mem);

// IN
BIORBD_API casadi_int libforward_dynamics_casadi_n_in(void);
BIORBD_API const char* libforward_dynamics_casadi_name_in(casadi_int i);
BIORBD_API const casadi_int* libforward_dynamics_casadi_sparsity_in(casadi_int i);

// OUT
BIORBD_API casadi_int libforward_dynamics_casadi_n_out(void);
BIORBD_API const char* libforward_dynamics_casadi_name_out(casadi_int i);
BIORBD_API const casadi_int* libforward_dynamics_casadi_sparsity_out(casadi_int i);

BIORBD_API int libforward_dynamics_casadi_work(casadi_int *sz_arg,
                                               casadi_int* sz_res,
                                               casadi_int *sz_iw,
                                               casadi_int *sz_w);


#ifdef __cplusplus
} /* extern "C" */
#endif

#endif // BIORBD_ACADO_INTERFACE_H
