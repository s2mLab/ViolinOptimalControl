#include "dynamics.h"
#include <rbdl/Dynamics.h>
#include "BiorbdModel.h"
#include "RigidBody/GeneralizedCoordinates.h"
#include "RigidBody/GeneralizedTorque.h"
#include "Muscles/StateDynamics.h"

void forwardDynamics_noContact(
        double *x,
        double *rhs,
        void *){
    dispatchQandQdot(x);

    if (nMus > 0) {
        dispatchActivation(x);
        Tau = m.muscularJointTorque(musclesStates, true, &Q, &Qdot);
    }
    else {
        Tau.setZero();
    }

    for(unsigned int i=0; i<nTau; ++i) {
        Tau[i] += x[nQ + nQdot + nMus + i];
    }

    forwardDynamics_noContact(Q, Qdot, Tau, rhs);
    validityCheck();
}

void forwardDynamics_noContact(
        const biorbd::rigidbody::GeneralizedCoordinates& Q,
        const biorbd::rigidbody::GeneralizedCoordinates& Qdot,
        const biorbd::rigidbody::GeneralizedTorque& Tau,
        double *rhs){
    RigidBodyDynamics::ForwardDynamics(m, Q, Qdot, Tau, Qddot);
    for(unsigned int i = 0; i<nQ; ++i) {
        rhs[i] = Qdot[i];
        rhs[i + nQdot] = Qddot[i];
    }
}

void forwardDynamics_contact(
        double *x,
        double *rhs,
        void *){
    dispatchQandQdot(x);

    if (nMus > 0) {
        dispatchActivation(x);
        Tau = m.muscularJointTorque(musclesStates, true, &Q, &Qdot);
    }
    else {
        Tau.setZero();
    }

    for(unsigned int i=0; i<nTau; ++i) {
        Tau[i] += x[nQ + nQdot + nMus + i];
    }

    forwardDynamics_contact(Q, Qdot, Tau, rhs);
    validityCheck();
}

void forwardDynamics_contact(
        const biorbd::rigidbody::GeneralizedCoordinates& Q,
        const biorbd::rigidbody::GeneralizedCoordinates& Qdot,
        const biorbd::rigidbody::GeneralizedTorque& Tau,
        double *rhs){
    RigidBodyDynamics::ConstraintSet& CS = m.getConstraints();
    RigidBodyDynamics::ForwardDynamicsConstraintsDirect(m, Q, Qdot, Tau, CS, Qddot);
    //RigidBodyDynamics::ForwardDynamicsContactsKokkevis(m, Q, Qdot, Tau, CS, Qddot);
    for(unsigned int i = 0; i<nQ; ++i){ // Assuming nQ == nQdot
        rhs[i] = Qdot[i];
        rhs[i + nQdot] = Qddot[i];
    }
}
