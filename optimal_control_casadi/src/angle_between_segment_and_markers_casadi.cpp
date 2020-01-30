#include "angle_between_segment_and_markers_casadi.h"

#ifdef __cplusplus
extern "C" {
#endif

// Declare sparsity dimensions
static bool isSparsityFilled(false);
static casadi_int Q_sparsity[3] = {-1, 1, 1};
static casadi_int UpdateKin_sparsity[3] = {-1, 1, 1};

static casadi_int SegmentIndexAndAxis_sparsity[3] = {-1, 1, 1};
static casadi_int MarkersIndex_sparsity[3] = {-1, 1, 1};
static casadi_int Angle_sparsity[3] = {-1, 1, 1};

void libangle_between_segment_and_markers_casadi_fillSparsity(){
    if (!isSparsityFilled){
        Q_sparsity[0] = m.nbQ() + m.nbQdot();
        UpdateKin_sparsity[0] = 1;

        SegmentIndexAndAxis_sparsity[0] = 2;
        MarkersIndex_sparsity[0] = 2;
        Angle_sparsity[0] = 1;

        isSparsityFilled = true;
    }
}

const char* libangle_between_segment_and_markers_casadi_name(void){
    return "libangle_between_segment_and_markers_casadi";
}

int libangle_between_segment_and_markers_casadi(
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
    if (updateKinematics){
        m.UpdateKinematicsCustom(&Q);
    }

    std::vector<biorbd::utils::Vector> axes;
    // Get the RT for the segment
    unsigned int segmentIdx = static_cast<unsigned int>(arg[2][0]);
    biorbd::utils::Rotation rt(m.globalJCS(segmentIdx).rot());

    // Extract the respective axes
    AXIS axisIdx = static_cast<AXIS>(arg[2][1]);
    if (axisIdx == AXIS::X){
        axes.push_back(rt.axe(0));
    } else if (axisIdx == AXIS::MINUS_X){
        axes.push_back(-rt.axe(0));
    } else if (axisIdx == AXIS::Y){
        axes.push_back(rt.axe(1));
    } else if (axisIdx == AXIS::MINUS_Y){
        axes.push_back(-rt.axe(1));
    } else if (axisIdx == AXIS::Z){
        axes.push_back(rt.axe(2));
    } else if (axisIdx == AXIS::MINUS_Z){
        axes.push_back(-rt.axe(2));
    }

    // Get the second axis by subtracting the two markers
    unsigned int idx1(static_cast<unsigned int>(arg[3][0]));
    unsigned int idx2(static_cast<unsigned int>(arg[3][1]));
    biorbd::rigidbody::NodeSegment M1(m.marker(Q, idx1, true, false));
    biorbd::rigidbody::NodeSegment M2(m.marker(Q, idx2, true, false));
    axes.push_back(M2 - M1);

    // Return the answers
    res[0][0] = axes[0].dot(axes[1]);

    return 0;
}

// IN
casadi_int libangle_between_segment_and_markers_casadi_n_in(void){
    return 4;
}
const char* libangle_between_segment_and_markers_casadi_name_in(casadi_int i){
    switch (i) {
    case 0: return "States";
    case 1: return "UpdateKinematics";
    case 2: return "SegmentIndexAndAxis";
    case 3: return "MarkersIndex";
    default: return nullptr;
    }
}
const casadi_int* libangle_between_segment_and_markers_casadi_sparsity_in(casadi_int i) {
    libangle_between_segment_and_markers_casadi_fillSparsity();
    switch (i) {
    case 0: return Q_sparsity;
    case 1: return UpdateKin_sparsity;
    case 2: return SegmentIndexAndAxis_sparsity;
    case 3: return MarkersIndex_sparsity;
    default: return nullptr;
    }
}

// OUT
casadi_int libangle_between_segment_and_markers_casadi_n_out(void){
    return 1;
}
const char* libangle_between_segment_and_markers_casadi_name_out(casadi_int i){
    switch (i) {
    case 0: return "Angle";
    default: return nullptr;
    }
}
const casadi_int* libangle_between_segment_and_markers_casadi_sparsity_out(casadi_int i) {
    libangle_between_segment_and_markers_casadi_fillSparsity();
    switch (i) {
    case 0: return Angle_sparsity;
    default: return nullptr;
    }
}

#ifdef __cplusplus
} /* extern "C" */
#endif
