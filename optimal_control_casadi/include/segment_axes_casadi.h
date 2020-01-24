#ifndef SEGMENT_AXES_CASADI_H
#define SEGMENT_AXES_CASADI_H

#include <math.h>

#include "biorbd.h"
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
void libsegment_axes_casadi_fillSparsity();

// Functions
BIORBD_API const char* libsegment_axes_casadi_name(void);
BIORBD_API int libsegment_axes_casadi(const casadi_real** arg,
                                                   casadi_real** res,
                                                   casadi_int* iw,
                                                   casadi_real* w,
                                                   void* mem);

// IN
BIORBD_API casadi_int libsegment_axes_casadi_n_in(void);
BIORBD_API const char* libsegment_axes_casadi_name_in(casadi_int i);
BIORBD_API const casadi_int* libsegment_axes_casadi_sparsity_in(casadi_int i);

// OUT
BIORBD_API casadi_int libsegment_axes_casadi_n_out(void);
BIORBD_API const char* libsegment_axes_casadi_name_out(casadi_int i);
BIORBD_API const casadi_int* libsegment_axes_casadi_sparsity_out(casadi_int i);

BIORBD_API int libsegment_axes_casadi_work(casadi_int *sz_arg,
                                               casadi_int* sz_res,
                                               casadi_int *sz_iw,
                                               casadi_int *sz_w);


#ifdef __cplusplus
} /* extern "C" */
#endif

#endif // SEGMENT_AXES_CASADI_H
