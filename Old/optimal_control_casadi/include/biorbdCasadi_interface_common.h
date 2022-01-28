#ifndef BIORBD_CASADI_INTERFACE_COMMON_H
#define BIORBD_CASADI_INTERFACE_COMMON_H

#include <math.h>
#include "biorbd.h"
extern biorbd::Model m;

enum AXIS{
    X,
    Y,
    Z,
    MINUS_X,
    MINUS_Y,
    MINUS_Z,
};

biorbd::utils::String getAxisInString(AXIS axis){
    if (axis == AXIS::X){
        return "x";
    } else if (axis == AXIS::Y){
        return "y";
    } else if (axis == AXIS::Z){
        return "z";
    } else {
        return "";
    }
}

enum ANGLE_SEQUENCE{
    XYZ,
    XZY,
    YXZ,
    YZX,
    ZXY,
    ZYX,
    NO_SEQUENCE
};

biorbd::utils::String getAngleSequenceInString(ANGLE_SEQUENCE sequence){
    if (sequence == ANGLE_SEQUENCE::XYZ){
        return "xyz";
    } else if (sequence == ANGLE_SEQUENCE::XZY){
        return "xzy";
    } else if (sequence == ANGLE_SEQUENCE::YXZ){
        return "yxz";
    } else if (sequence == ANGLE_SEQUENCE::YZX){
        return "yzx";
    } else if (sequence == ANGLE_SEQUENCE::ZXY){
        return "zxy";
    } else if (sequence == ANGLE_SEQUENCE::ZYX){
        return "zyx";
    } else {
        return "";
    }
}

enum PLANE{
    XY,
    YZ,
    XZ,
    NO_PLANE
};

biorbd::utils::String getPlaneInString(PLANE plane){
    if (plane == PLANE::XY){
        return "xy";
    } else if (plane == PLANE::YZ){
        return "yz";
    } else if (plane == PLANE::XZ){
        return "xz";
    } else {
        return "";
    }
}

enum ViolinStringNames{
    E,
    A,
    D,
    G,
    NO_STRING
};

biorbd::utils::String getViolinStringInString(ViolinStringNames name){
    if (name == ViolinStringNames::E){
        return "E";
    } else if (name == ViolinStringNames::A){
        return "A";
    } else if (name == ViolinStringNames::D){
        return "D";
    } else if (name == ViolinStringNames::G){
        return "G";
    } else {
        return "";
    }
}

enum ODE_SOLVER{
    COLLOCATION,
    RK,
    CVODES,
    NO_SOLVER
};



#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int long long int
#endif

#endif // BIORBD_CASADI_INTERFACE_COMMON_H
