#ifndef __DYNAMICS_H
#define __DYNAMICS_H
#include "s2mMusculoSkeletalModel.h"


//#define CHECK_MAX_FORCE

void forwardDynamics(const s2mGenCoord& Q, const s2mGenCoord& Qdot, const s2mTau& Tau, double *rhs);
void forwardDynamicsFromMuscleActivation( double *x, double *rhs, void *user_data);
void forwardDynamicsFromJointTorque( double *x, double *rhs, void *user_data);
void forwardDynamicsFromMuscleActivationAndTorque(double *x, double *rhs, void *);
void forwardDynamicsMultiStage( double *x, double *rhs, void *user_data);
void forwardDynamicsFromMuscleActivationAndTorqueContact( double *x, double *rhs, void *user_data);
void forwardDynamicsFromTorqueContact( double *x, double *rhs, void *user_data);

extern s2mMusculoSkeletalModel m;
extern unsigned int nQ;
extern unsigned int nQdot;
extern unsigned int nTau;
extern unsigned int nTags;
extern unsigned int nMus;
extern unsigned int nPhases;

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
