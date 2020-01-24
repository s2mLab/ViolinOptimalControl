#include "forward_kinematics_casadi.h"

#ifdef __cplusplus
extern "C" {
#endif

// Declare sparsity dimensions
static bool isSparsityFilled(false);
static casadi_int Q_sparsity[3] = {-1, 1, 1};
static casadi_int UpdateKin_sparsity[3] = {-1, 1, 1};

static casadi_int MarkerIndex_sparsity[3] = {-1, 1, 1}; // idx for forward kin
static casadi_int Marker_sparsity[3] = {-1, 1, 1}; // output for forward kin

void libforward_kinematics_casadi_fillSparsity(){
    if (!isSparsityFilled){
        Q_sparsity[0] = m.nbQ() + m.nbQdot();
        UpdateKin_sparsity[0] = 1;

        MarkerIndex_sparsity[0] = 1;
        Marker_sparsity[0] = 3;

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

    // Dispatch data
    biorbd::rigidbody::GeneralizedCoordinates Q(m);
    for (unsigned int i = 0; i < m.nbQ(); ++i){
        Q[i] = arg[0][i];
    }
    bool updateKinematics = static_cast<bool>(arg[1][0]);

    // Perform the forward kinematics
    unsigned int markerIdx = static_cast<unsigned int>(arg[2][0]);
    biorbd::rigidbody::NodeSegment marker(m.marker(Q, markerIdx, true, updateKinematics));

    // Return the answers
    for (unsigned int i=0; i<3; ++i){
        res[0][i] = marker[i];
    }

    return 0;
}

// IN
casadi_int libforward_kinematics_casadi_n_in(void){
    return 3;
}
const char* libforward_kinematics_casadi_name_in(casadi_int i){
    switch (i) {
    case 0: return "States";
    case 1: return "UpdateKinematics";
    case 2: return "MarkerIndex";
    default: return nullptr;
    }
}
const casadi_int* libforward_kinematics_casadi_sparsity_in(casadi_int i) {
    libforward_kinematics_casadi_fillSparsity();
    switch (i) {
    case 0: return Q_sparsity;
    case 1: return UpdateKin_sparsity;
    case 2: return MarkerIndex_sparsity;
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
