#ifndef VIOLIN_OPTIMIZATION_DYNAMICS_H
#define VIOLIN_OPTIMIZATION_DYNAMICS_H
#include "biorbd_declarer.h"
#include "utils.h"

void forwardDynamics_noContact(
        double *x,
        double *rhs,
        void *user_data);
void forwardDynamics_noContact(
        const biorbd::rigidbody::GeneralizedCoordinates& Q,
        const biorbd::rigidbody::GeneralizedCoordinates& Qdot,
        const biorbd::rigidbody::GeneralizedTorque& Tau,
        double *rhs);
void forwardDynamics_contact(
        double *x,
        double *rhs,
        void *user_data);
void forwardDynamics_contact(
        const biorbd::rigidbody::GeneralizedCoordinates& Q,
        const biorbd::rigidbody::GeneralizedCoordinates& Qdot,
        const biorbd::rigidbody::GeneralizedTorque& Tau,
        double *rhs);

// Show STL vector
template<typename T>
std::ostream& operator<< (std::ostream& out, const std::vector<T>& v) {
    out << "[";
    size_t last = v.size() - 1;
    for(size_t i = 0; i < v.size(); ++i) {
        out << v[i];
        if (i != last)
            out << ", ";
    }
    out << "]";
    return out;
}


#endif  // VIOLIN_OPTIMIZATION_DYNAMICS_H
