#ifndef BIORBD_CASADI_INTERFACE_COMMON_H
#define BIORBD_CASADI_INTERFACE_COMMON_H

#include <math.h>
#include "biorbd.h"
extern biorbd::Model m;

#ifndef AXIS_DEF
#define AXIS_DEF
enum AXIS{
    X,
    Y,
    Z,
    MINUS_X,
    MINUS_Y,
    MINUS_Z,
};

biorbd::utils::String getAxisInString(AXIS axis){
    biorbd::utils::String axis_out;
    if (axis == AXIS::X){
        axis_out = "x";
    } else if (axis == AXIS::Y){
        axis_out = "y";
    } else if (axis == AXIS::Z){
        axis_out = "z";
    }
    return axis_out;
}
#endif


#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int long long int
#endif

#endif // BIORBD_CASADI_INTERFACE_COMMON_H
