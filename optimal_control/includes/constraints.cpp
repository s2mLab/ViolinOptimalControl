#include "constraints.h"

#ifndef PI
#define PI 3.141592
#endif

#include "RigidBody/GeneralizedCoordinates.h"
#include "RigidBody/GeneralizedTorque.h"
#include "RigidBody/NodeBone.h"
#include "Muscles/StateDynamics.h"

// Preallocate the variables
static biorbd::rigidbody::NodeBone tag;

void statesZero( double *x, double *g, void * ){
    for(unsigned int i=0; i<nQ + nQdot; ++i) {
        g[i] =  x[i];
    }
}

void velocityZero( double *x, double *g, void * ){
    for(unsigned int i=0; i<nQdot; ++i) {
        g[i] =  x[nQ+i];
    }
}

void activationsZero( double *x, double *g, void * ){
    for(unsigned int i=0; i<nMus; ++i) {
        g[i] =  x[i+nQ+nQdot];
    }
}

void torquesZero( double *x, double *g, void * ){
    for(unsigned int i=0; i<nTau; ++i) {
        g[i] =  x[i+nQ+nQdot+nMus];
    }
}

void rotbras( double *x, double *g, void * ){
    g[0] = x[nQ-1]-PI/4;
}

void violonUp( double *x, double *g, void * ){
    g[0] = x[1]+1.13;       // shoulder at 65° in abduction (Arm_RotX = -1.13)
    g[1] = x[2]-0.61;       // shoulder at 35° in flexion (Arm_RotZ = 0.61)
    g[2] = x[3]+0.35;       // shoulder at 20°  (Arm_RotY = -0.35)
    g[3] = x[4]-1.55;       // elbow at 110° (LowerArm1_RotZ = 1.55)
}

void violonDown( double *x, double *g, void * ){
    g[0] = x[1]+0.70;       // shoulder at 40° in  (Arm_RotX = -0.70)
    g[1] = x[2]-0.17;       // shoulder at 10° in flexion (Arm_RotZ = 0.17)
    g[2] = x[3];             // shoulder at 0°  (Arm_RotY = 0.0)
    g[3] = x[4]-0.59;       // elbow at 35° (LowerArm1_RotZ = 0.61)
}

void markerPosition(double *x, double *g, void *user_data ){
    unsigned int numTag = static_cast<unsigned int*>(user_data)[0];

    for(unsigned int i = 0; i<nQ; ++i){
        Q[i] = x[i];
    }
    tag = m.marker(Q, numTag, true, true);

    g[0]=tag[0];
    g[1]=tag[1];
    g[2]=tag[2];
}

void forceConstraintFromMuscleActivation( double *x, double *g, void *user_data){
//    forwardDynamicsFromMuscleActivationAndTorqueContact(x, g, user_data);

    // Dispatch the inputs
    for(unsigned int i = 0; i<nQ; ++i){
        Q[i] = x[i];
        Qdot[i] = x[i+nQ];
    }
    m.updateMuscles(Q, Qdot, true);


    for(unsigned int i = 0; i<nMus; ++i){
        state[i] = std::make_shared<biorbd::muscles::StateDynamics>(0, x[i+nQ+nQdot]);
    }

    // Compute the torques from muscles
    Tau = m.muscularJointTorque(state, false, &Q, &Qdot);
    for(unsigned int i=0; i<nTau; ++i){
        Tau[i] += x[i+nQ+nQdot+nMus];
        //std::cout<<"Torques additionnels:"<<x[i+nQ+nQdot+nMus]<<std::endl;
    }
    // Compute the forward dynamics
    RigidBodyDynamics::ConstraintSet& CS = m.getConstraints();
    RigidBodyDynamics::ForwardDynamicsConstraintsDirect(m, Q, Qdot, Tau, CS, Qddot);
    g[0]=CS.force(0);
    g[1]=CS.force(1);

}

void forceConstraintFromTorque(double *x, double *g, void *user_data)
{
        // Dispatch the inputs
        for(unsigned int i = 0; i<nQ; ++i){
            Q[i] = x[i];
            Qdot[i] = x[i+nQ];
            Tau[i]= x[i+nQ+nQdot];
        }
        // Compute the forward dynamics
        RigidBodyDynamics::ConstraintSet& CS = m.getConstraints();
        RigidBodyDynamics::ForwardDynamicsConstraintsDirect(m, Q, Qdot, Tau, CS, Qddot);

        g[0]=CS.force(0);
        g[1]=CS.force(1);

}
