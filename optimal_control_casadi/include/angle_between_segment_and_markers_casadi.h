#ifndef ANGLE_BETWEEN_SEGMENT_AND_MARKERS_CASADI_H
#define ANGLE_BETWEEN_SEGMENT_AND_MARKERS_CASADI_H

#include <math.h>

#include "biorbdCasadi_interface_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Preparing stuff
void libangle_between_segment_and_markers_casadi_fillSparsity();

// Functions
BIORBD_API const char* libangle_between_segment_and_markers_casadi_name(void);
BIORBD_API int libangle_between_segment_and_markers_casadi(const casadi_real** arg,
                                                   casadi_real** res,
                                                   casadi_int* iw,
                                                   casadi_real* w,
                                                   void* mem);

// IN
BIORBD_API casadi_int libangle_between_segment_and_markers_casadi_n_in(void);
BIORBD_API const char* libangle_between_segment_and_markers_casadi_name_in(casadi_int i);
BIORBD_API const casadi_int* libangle_between_segment_and_markers_casadi_sparsity_in(casadi_int i);

// OUT
BIORBD_API casadi_int libangle_between_segment_and_markers_casadi_n_out(void);
BIORBD_API const char* libangle_between_segment_and_markers_casadi_name_out(casadi_int i);
BIORBD_API const casadi_int* libangle_between_segment_and_markers_casadi_sparsity_out(casadi_int i);

BIORBD_API int libangle_between_segment_and_markers_casadi_work(casadi_int *sz_arg,
                                               casadi_int* sz_res,
                                               casadi_int *sz_iw,
                                               casadi_int *sz_w);


#ifdef __cplusplus
} /* extern "C" */
#endif

#endif // ANGLE_BETWEEN_SEGMENT_AND_MARKERS_CASADI_H
