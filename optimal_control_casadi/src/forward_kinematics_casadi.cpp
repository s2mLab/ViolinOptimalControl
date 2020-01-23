#include "forward_kinematics_casadi.h"

#ifdef __cplusplus
extern "C" {
#endif

// Declare sparsity dimensions
static bool isSparsityFilled(false);
static casadi_int Index_sparsity[3] = {-1, 1, 1};
static casadi_int Marker_sparsity[3] = {-1, 1, 1};
static casadi_int Q_sparsity[3] = {-1, 1, 1};

void libforward_kinematics_casadi_fillSparsity(){
    if (!isSparsityFilled){
        Index_sparsity[0] = 1;
        Marker_sparsity[0] = 3;
        Q_sparsity[0] = m.nbQ() + m.nbQdot();
        isSparsityFilled = true;
    }
}

const char* libforward_kinematics_casadi_name(void){
    return "libforward_kinematics_casadi";
}

int libforward_kinematics_casadi(
        const casadi_real** arg,
        casadi_real** res,
        casadi_int*,
        casadi_real*, void*){
    biorbd::rigidbody::GeneralizedCoordinates Q(m);
    biorbd::rigidbody::GeneralizedTorque Tau(m);

    // Dispatch data
    unsigned int markerIdx = static_cast<unsigned int>(arg[1][0]);
    for (unsigned int i = 0; i < m.nbQ(); ++i){
        Q[i] = arg[0][i];
    }

    // Perform the forward kinematics
    biorbd::rigidbody::NodeSegment marker(m.marker(Q, markerIdx));

    // Return the answers
    for (unsigned int i=0; i<3; ++i){
        res[0][i] = marker[i];
    }

    return 0;
}

// IN
casadi_int libforward_kinematics_casadi_n_in(void){
    return 2;
}
const char* libforward_kinematics_casadi_name_in(casadi_int i){
    switch (i) {
    case 0: return "States";
    case 1: return "MarkerIndex";
    default: return nullptr;
    }
}
const casadi_int* libforward_kinematics_casadi_sparsity_in(casadi_int i) {
    libforward_kinematics_casadi_fillSparsity();
    switch (i) {
    case 0: return Q_sparsity;
    case 1: return Index_sparsity;
    default: return nullptr;
    }
}

// OUT
casadi_int libforward_kinematics_casadi_n_out(void){
    return 1;
}
const char* libforward_kinematics_casadi_name_out(casadi_int i){
    switch (i) {
        case 0: return "Marker";
        default: return nullptr;
    }
}
const casadi_int* libforward_kinematics_casadi_sparsity_out(casadi_int i) {
    libforward_kinematics_casadi_fillSparsity();
    switch (i) {
        case 0: return Marker_sparsity;
    default: return nullptr;
    }
}

int libforward_kinematics_casadi_work(casadi_int *sz_arg,
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
