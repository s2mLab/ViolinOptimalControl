#ifndef __DYNAMICS_H
#define __DYNAMICS_H
#include "s2mMusculoSkeletalModel.h"

void forwardDynamics(const s2mGenCoord& Q, const s2mGenCoord& Qdot, const s2mTau& Tau, double *rhs);
void forwardDynamicsFromMuscleActivation( double *x, double *rhs, void *user_data);
void forwardDynamicsFromJointTorque( double *x, double *rhs, void *user_data);

extern s2mMusculoSkeletalModel m;
extern unsigned int nQ;
extern unsigned int nQdot;
extern unsigned int nTau;
extern unsigned int nTags;
extern unsigned int nMus;

#endif
