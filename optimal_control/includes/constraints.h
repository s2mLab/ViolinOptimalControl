#ifndef __CONSTRAINTS_H
#define __CONSTRAINTS_H
#include "biorbd/BiorbdModel.h"
#include "RigidBody/GeneralizedCoordinates.h"
#include "RigidBody/GeneralizedTorque.h"
#include "Muscles/StateDynamics.h"
#include "dynamics.h"
#include "Utils/RotoTrans.h"
#include "RigidBody/NodeBone.h"

void statesZero( double *x, double *g, void * );
void velocityZero( double *x, double *g, void * );
void activationsZero( double *x, double *g, void * );
void torquesZero( double *x, double *g, void * );
void rotbras( double *x, double *g, void * );
void violonUp( double *x, double *g, void * );
void violonDown( double *x, double *g, void * );
void markerPosition(double *x, double *g, void *user_data );
void forceConstraintFromMuscleActivation( double *x, double *g, void *user_data);
void forceConstraintFromTorque(double *x, double *g, void *user_data);
void orthogonalProjected(double *x, double *g, void *user_data);

extern biorbd::Model m;
extern unsigned int nQ;
extern unsigned int nQdot;
extern unsigned int nTau;
extern unsigned int nMarkers;
extern unsigned int nMus;
extern unsigned int nPhases;
extern GeneralizedCoordinates Q, Qdot, Qddot;
extern GeneralizedTorque Tau;
extern std::vector<std::shared_ptr<biorbd::muscles::StateDynamics>> musclesStates; // controls


#endif
