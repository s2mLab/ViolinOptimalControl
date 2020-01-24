#include "projectionOnSegment_casadi.h"

#ifdef __cplusplus
extern "C" {
#endif

// Declare sparsity dimensions
static bool isSparsityFilled(false);
static casadi_int Q_sparsity[3] = {-1, 1, 1};
static casadi_int UpdateKin_sparsity[3] = {-1, 1, 1};

static casadi_int MarkerToProjectIndex_sparsity[3] = {-1, 1, 1};
static casadi_int SegmentToProjectOnIndex_sparsity[3] = {-1, 1, 1};
static casadi_int ProjectedMarker_sparsity[3] = {-1, 1, 1}; // output for forward kin

void libprojectionOnSegment_casadi_fillSparsity(){
    if (!isSparsityFilled){
        Q_sparsity[0] = m.nbQ() + m.nbQdot();
        UpdateKin_sparsity[0] = 1;

        MarkerToProjectIndex_sparsity[0] = 1;
        SegmentToProjectOnIndex_sparsity[0] = 1;
        ProjectedMarker_sparsity[0] = 3;

        isSparsityFilled = true;
    }
}

const char* libprojectionOnSegment_casadi_name(void){
    return "libprojectionOnSegment_casadi";
}

int libprojectionOnSegment_casadi(
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

    // Project marker on the RT of a specific segment
    unsigned int segmentToProjectOnIdx = static_cast<unsigned int>(arg[2][0]);
    unsigned int markerToProjectIdx = static_cast<unsigned int>(arg[3][0]);
    biorbd::utils::RotoTrans rt(m.globalJCS(Q, segmentToProjectOnIdx));
    biorbd::rigidbody::NodeSegment markerToProject(m.marker(Q, markerToProjectIdx, false, updateKinematics));
    markerToProject.applyRT(rt.transpose());

    // Return the answers
    for (unsigned int i=0; i<3; ++i){
        res[0][i] = markerToProject[i];
    }
    return 0;
}

// IN
casadi_int libprojectionOnSegment_casadi_n_in(void){
    return 4;
}
const char* libprojectionOnSegment_casadi_name_in(casadi_int i){
    switch (i) {
    case 0: return "States";
    case 1: return "UpdateKinematics";
    case 2: return "SegmentToProjectOnIndex";
    case 3: return "MarkerToProjectIndex";
    default: return nullptr;
    }
}
const casadi_int* libprojectionOnSegment_casadi_sparsity_in(casadi_int i) {
    libprojectionOnSegment_casadi_fillSparsity();
    switch (i) {
    case 0: return Q_sparsity;
    case 1: return UpdateKin_sparsity;
    case 2: return SegmentToProjectOnIndex_sparsity;
    case 3: return MarkerToProjectIndex_sparsity;
    default: return nullptr;
    }
}

// OUT
casadi_int libprojectionOnSegment_casadi_n_out(void){
    return 1;
}
const char* libprojectionOnSegment_casadi_name_out(casadi_int i){
    switch (i) {
        case 0: return "ProjectedMarker";
        default: return nullptr;
    }
}
const casadi_int* libprojectionOnSegment_casadi_sparsity_out(casadi_int i) {
    libprojectionOnSegment_casadi_fillSparsity();
    switch (i) {
        case 0: return ProjectedMarker_sparsity;
    default: return nullptr;
    }
}

int libprojectionOnSegment_casadi_work(casadi_int *sz_arg,
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
