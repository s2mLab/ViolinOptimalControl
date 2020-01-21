#ifndef VIOLIN_OPTIMIZATION_BIORBD_DECLARER_H
#define VIOLIN_OPTIMIZATION_BIORBD_DECLARER_H

#include "biorbd.h"

extern biorbd::Model m;
extern unsigned int nQ;
extern unsigned int nQdot;
extern unsigned int nTau;
extern unsigned int nMus;
extern unsigned int nMarkers;

extern biorbd::rigidbody::GeneralizedCoordinates Q;
extern biorbd::rigidbody::GeneralizedVelocity Qdot;
extern biorbd::rigidbody::GeneralizedAcceleration Qddot;
extern biorbd::rigidbody::GeneralizedTorque Tau;
extern std::vector<std::shared_ptr<biorbd::muscles::StateDynamics>> musclesStates;

//#define CHECK_MAX_FORCE
//#define CHECK_FORCE_IF_LOW_ACTIVATION
//#define CHECK_MUSCLE_LENGTH_IS_POSITIVE

#endif  // VIOLIN_OPTIMIZATION_BIORBD_DECLARER_H
