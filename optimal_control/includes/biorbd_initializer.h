#ifndef VIOLIN_OPTIMIZATION_BIORBD_INITIALISER_H
#define VIOLIN_OPTIMIZATION_BIORBD_INITIALISER_H

#include "biorbd_declarer.h"

unsigned int nQ(m.nbQ());
unsigned int nQdot(m.nbQdot());
unsigned int nTau(m.nbGeneralizedTorque());
unsigned int nMus(m.nbMuscleTotal());
unsigned int nMarkers(m.nMarkers());

biorbd::rigidbody::GeneralizedCoordinates Q(m);
biorbd::rigidbody::GeneralizedCoordinates Qdot(m);
biorbd::rigidbody::GeneralizedCoordinates Qddot(m);
biorbd::rigidbody::GeneralizedTorque Tau(m);
std::vector<std::shared_ptr<biorbd::muscles::StateDynamics>> musclesStates(nMus);

#endif  // VIOLIN_OPTIMIZATION_BIORBD_INITIALISER_H
