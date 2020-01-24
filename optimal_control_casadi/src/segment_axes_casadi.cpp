#include "segment_axes_casadi.h"

#ifdef __cplusplus
extern "C" {
#endif

// Declare sparsity dimensions
static bool isSparsityFilled(false);
static casadi_int Q_sparsity[3] = {-1, 1, 1};
static casadi_int UpdateKin_sparsity[3] = {-1, 1, 1};

static casadi_int Segment1Index_sparsity[3] = {-1, 1, 1};
static casadi_int Segment1Axis_sparsity[3] = {-1, 1, 1};
static casadi_int Segment2Index_sparsity[3] = {-1, 1, 1};
static casadi_int Segment2Axis_sparsity[3] = {-1, 1, 1};
static casadi_int Axis_sparsity[3] = {-1, 1, 1};

void libsegment_axes_casadi_fillSparsity(){
    if (!isSparsityFilled){
        Q_sparsity[0] = m.nbQ() + m.nbQdot();
        UpdateKin_sparsity[0] = 1;

        Segment1Index_sparsity[0] = 1;
        Segment1Axis_sparsity[0] = 1;
        Segment2Index_sparsity[0] = 1;
        Segment2Axis_sparsity[0] = 1;
        Axis_sparsity[0] = 1;

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
    unsigned int segment1Idx = static_cast<unsigned int>(arg[2][0]);
    unsigned int segment2Idx = static_cast<unsigned int>(arg[4][0]);
    biorbd::utils::Rotation rt1(m.globalJCS(Q, segment1Idx).rot());
    biorbd::utils::Rotation rt2(m.globalJCS(Q, segment2Idx).rot());

    // Extract the respective axes
    AXIS axis1Idx = static_cast<AXIS>(arg[3][0]);
    AXIS axis2Idx = static_cast<AXIS>(arg[5][0]);

    biorbd::utils::Vector axis1;
    if (axis1Idx == AXIS::X){
        axis1 = rt1.axe(0);
    } else if (axis1Idx == AXIS::MINUS_X){
        axis1 = -rt1.axe(0);
    } else if (axis1Idx == AXIS::Y){
        axis1 = rt1.axe(1);
    } else if (axis1Idx == AXIS::MINUS_Y){
        axis1 = -rt1.axe(1);
    } else if (axis1Idx == AXIS::Z){
        axis1 = rt1.axe(2);
    } else if (axis1Idx == AXIS::MINUS_Z){
        axis1 = -rt1.axe(2);
    }

    biorbd::utils::Vector axis2;
    if (axis2Idx == AXIS::X){
        axis2 = rt2.axe(0);
    } else if (axis2Idx == AXIS::MINUS_X){
        axis2 = -rt2.axe(0);
    } else if (axis2Idx == AXIS::Y){
        axis2 = rt2.axe(1);
    } else if (axis2Idx == AXIS::MINUS_Y){
        axis2 = -rt2.axe(1);
    } else if (axis2Idx == AXIS::Z){
        axis2 = rt2.axe(2);
    } else if (axis2Idx == AXIS::MINUS_Z){
        axis2 = -rt2.axe(2);
    }

    // Return the answers
    res[0][0] = 1 - axis1.dot(axis2);
//    res[0][0] = (axis1 - axis2)[0];
//    res[0][1] = (axis1 - axis2)[1];
//    res[0][2] = (axis1 - axis2)[2];

    return 0;
}

// IN
casadi_int libsegment_axes_casadi_n_in(void){
    return 6;
}
const char* libsegment_axes_casadi_name_in(casadi_int i){
    switch (i) {
    case 0: return "States";
    case 1: return "UpdateKinematics";
    case 2: return "Segment1Index";
    case 3: return "Segment1Axis";
    case 4: return "Segment2Index";
    case 5: return "Segment2Axis";
    default: return nullptr;
    }
}
const casadi_int* libsegment_axes_casadi_sparsity_in(casadi_int i) {
    libsegment_axes_casadi_fillSparsity();
    switch (i) {
    case 0: return Q_sparsity;
    case 1: return UpdateKin_sparsity;
    case 2: return Segment1Index_sparsity;
    case 3: return Segment1Axis_sparsity;
    case 4: return Segment2Index_sparsity;
    case 5: return Segment2Axis_sparsity;
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
