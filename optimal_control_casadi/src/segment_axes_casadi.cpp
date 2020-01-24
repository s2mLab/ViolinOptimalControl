#include "segment_axes_casadi.h"

#ifdef __cplusplus
extern "C" {
#endif

// Declare sparsity dimensions
static bool isSparsityFilled(false);
static casadi_int Q_sparsity[3] = {-1, 1, 1};
static casadi_int UpdateKin_sparsity[3] = {-1, 1, 1};

static casadi_int SegmentIndex_sparsity[3] = {-1, 1, 1};
static casadi_int Axis_sparsity[3] = {-1, 1, 1};

void libsegment_axes_casadi_fillSparsity(){
    if (!isSparsityFilled){
        Q_sparsity[0] = m.nbQ() + m.nbQdot();
        UpdateKin_sparsity[0] = 1;

        SegmentIndex_sparsity[0] = 1;
        Axis_sparsity[0] = 9;

        isSparsityFilled = true;
    }
}

const char* libsegment_axes_casadi_name(void){
    return "libsegment_axes_casadi";
}

int libsegment_axes_casadi(
        const casadi_real** arg,
        casadi_real** res,
        casadi_int*,
        casadi_real*, void*){

    // Dispatch data
    biorbd::rigidbody::GeneralizedCoordinates Q(m);
    for (unsigned int i = 0; i < m.nbQ(); ++i){
        Q[i] = arg[0][i];
    }
    bool updateKinematics = static_cast<bool>(arg[1][0]);

    // Get the RT for the segment
    unsigned int segmentIdx = static_cast<unsigned int>(arg[2][0]);
    biorbd::utils::Rotation rt(m.globalJCS(Q, segmentIdx).rot());

    // Return the answers
    for (unsigned int i=0; i<3; ++i){
        for (unsigned int j=0; j<3; ++j){
            res[0][i*3+j] = rt(i, j);
        }
    }

    return 0;
}

// IN
casadi_int libsegment_axes_casadi_n_in(void){
    return 3;
}
const char* libsegment_axes_casadi_name_in(casadi_int i){
    switch (i) {
    case 0: return "States";
    case 1: return "UpdateKinematics";
    case 2: return "SegmentIndex";
    default: return nullptr;
    }
}
const casadi_int* libsegment_axes_casadi_sparsity_in(casadi_int i) {
    libsegment_axes_casadi_fillSparsity();
    switch (i) {
    case 0: return Q_sparsity;
    case 1: return UpdateKin_sparsity;
    case 2: return SegmentIndex_sparsity;
    default: return nullptr;
    }
}

// OUT
casadi_int libsegment_axes_casadi_n_out(void){
    return 1;
}
const char* libsegment_axes_casadi_name_out(casadi_int i){
    switch (i) {
    case 0: return "Axes";
    default: return nullptr;
    }
}
const casadi_int* libsegment_axes_casadi_sparsity_out(casadi_int i) {
    libsegment_axes_casadi_fillSparsity();
    switch (i) {
    case 0: return Axis_sparsity;
    default: return nullptr;
    }
}

int libsegment_axes_casadi_work(casadi_int *sz_arg,
                                               casadi_int* sz_res,
                                               casadi_int *sz_iw,
                                               casadi_int *sz_w) {
    if (sz_arg) *sz_arg = m.nbQ();
    if (sz_res) *sz_res = m.nbMarkers();
    if (sz_iw) *sz_iw = 0;
    if (sz_w) *sz_w = 0;
    return 0;
}

#ifdef __cplusplus
} /* extern "C" */
#endif
