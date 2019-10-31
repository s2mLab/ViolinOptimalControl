#ifndef __DYNAMICS_H
#define __DYNAMICS_H
#include "biorbd/BiorbdModel.h"
#include "RigidBody/GeneralizedCoordinates.h"
#include "RigidBody/GeneralizedTorque.h"
#include "Muscles/StateDynamics.h"
using namespace biorbd::rigidbody;


//#define CHECK_MAX_FORCE
void forwardDynamics(const GeneralizedCoordinates& Q, const GeneralizedCoordinates& Qdot, const GeneralizedTorque& Tau, double *rhs);
void forwardDynamicsFromMuscleActivation( double *x, double *rhs, void *user_data);
void forwardDynamicsFromJointTorque( double *x, double *rhs, void *user_data);
void forwardDynamicsFromMuscleActivationAndTorque(double *x, double *rhs, void *);
void forwardDynamicsMultiStage( double *x, double *rhs, void *user_data);
void forwardDynamicsFromMuscleActivationAndTorqueContact( double *x, double *rhs, void *user_data);
void forwardDynamicsFromTorqueContact( double *x, double *rhs, void *user_data);

extern biorbd::Model m;
extern unsigned int nQ;
extern unsigned int nQdot;
extern unsigned int nTau;
extern unsigned int nMarkers;
extern unsigned int nMus;
extern unsigned int nPhases;
extern GeneralizedCoordinates Q, Qdot, Qddot;
extern GeneralizedTorque Tau;
extern std::vector<std::shared_ptr<biorbd::muscles::StateDynamics>> musclesStates;

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


#endif
