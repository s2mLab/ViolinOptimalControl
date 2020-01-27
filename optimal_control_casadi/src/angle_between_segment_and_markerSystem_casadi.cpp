#include "angle_between_segment_and_markerSystem_casadi.h"

#ifdef __cplusplus
extern "C" {
#endif

// Declare sparsity dimensions
static bool isSparsityFilled(false);
static casadi_int Q_sparsity[3] = {-1, 1, 1};
static casadi_int UpdateKin_sparsity[3] = {-1, 1, 1};

static casadi_int SegmentIndex_sparsity[3] = {-1, 1, 1};
static casadi_int AxisDescription_sparsity[3] = {-1, 1, 1};
static casadi_int AxisToRecalculate_sparsity[3] = {-1, 1, 1};
static casadi_int Angles_sparsity[3] = {-1, 1, 1};

void libangle_between_segment_and_markerSystem_casadi_fillSparsity(){
    if (!isSparsityFilled){
        Q_sparsity[0] = m.nbQ() + m.nbQdot();
        UpdateKin_sparsity[0] = 1;

        SegmentIndex_sparsity[0] = 2;
        AxisDescription_sparsity[0] = 3;
        AxisToRecalculate_sparsity[0] = 1;
        Angles_sparsity[0] = 3;

        isSparsityFilled = true;
    }
}

const char* libangle_between_segment_and_markerSystem_casadi_name(void){
    return "libangle_between_segment_and_markerSystem_casadi";
}



int libangle_between_segment_and_markerSystem_casadi(
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

    // Get the system of axes of the segment to align
    biorbd::utils::Rotation r_seg(
                m.globalJCS(static_cast<unsigned int>(arg[3][0])).rot());

    // Get the system of axes from the markers
    biorbd::utils::String axis1name(
                getAxisInString(static_cast<AXIS>(arg[4][0])));
    biorbd::rigidbody::NodeSegment axis1Beg(
                m.marker(Q, static_cast<unsigned int>(arg[4][1]), true, false));
    biorbd::rigidbody::NodeSegment axis1End(
                m.marker(Q, static_cast<unsigned int>(arg[4][2]), true, false));

    biorbd::utils::String axis2name(
                getAxisInString(static_cast<AXIS>(arg[5][0])));
    biorbd::rigidbody::NodeSegment axis2Beg(
                m.marker(Q, static_cast<unsigned int>(arg[5][1]), true, false));
    biorbd::rigidbody::NodeSegment axis2End(
                m.marker(Q, static_cast<unsigned int>(arg[5][2]), true, false));

    biorbd::utils::String axisToRecalculate(
                getAxisInString(static_cast<AXIS>(arg[6][0])));

    biorbd::utils::Rotation r_markers(
                biorbd::utils::Rotation::fromMarkers(
                    {axis1Beg, axis1End}, {axis2Beg, axis2End}, {axis1name, axis2name},
                    axisToRecalculate));

    // Get the angle between the two reference frames
    biorbd::utils::Rotation r(r_seg.transpose() * r_markers);
    biorbd::utils::Vector angles(biorbd::utils::Rotation::toEulerAngles(r, "xyz"));

    // Return the answers
    for (unsigned int i=0; i<angles.size(); ++i){
        res[0][i] = angles(i);
    }

    return 0;
}

// IN
casadi_int libangle_between_segment_and_markerSystem_casadi_n_in(void){
    return 6;
}
const char* libangle_between_segment_and_markerSystem_casadi_name_in(casadi_int i){
    switch (i) {
    case 0: return "States";
    case 1: return "UpdateKinematics";
    case 2: return "SegmentIndex";
    case 3: return "Axis1Description";
    case 4: return "Axis2Description";
    case 5: return "AxisToRecalculate";
    default: return nullptr;
    }
}
const casadi_int* libangle_between_segment_and_markerSystem_casadi_sparsity_in(casadi_int i) {
    libangle_between_segment_and_markerSystem_casadi_fillSparsity();
    switch (i) {
    case 0: return Q_sparsity;
    case 1: return UpdateKin_sparsity;
    case 2: return SegmentIndex_sparsity;
    case 3: return AxisDescription_sparsity;
    case 4: return AxisDescription_sparsity;
    case 5: return AxisToRecalculate_sparsity;
    default: return nullptr;
    }
}

// OUT
casadi_int libangle_between_segment_and_markerSystem_casadi_n_out(void){
    return 1;
}
const char* libangle_between_segment_and_markerSystem_casadi_name_out(casadi_int i){
    switch (i) {
    case 0: return "Angles";
    default: return nullptr;
    }
}
const casadi_int* libangle_between_segment_and_markerSystem_casadi_sparsity_out(casadi_int i) {
    libangle_between_segment_and_markerSystem_casadi_fillSparsity();
    switch (i) {
    case 0: return Angles_sparsity;
    default: return nullptr;
    }
}

#ifdef __cplusplus
} /* extern "C" */
#endif
