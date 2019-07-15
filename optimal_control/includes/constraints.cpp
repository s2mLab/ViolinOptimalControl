#include "constraints.h"


void statesZero( double *x, double *g, void * ){
    for (unsigned int i =0; i<nQ + nQdot; ++i) {
        g[i] =  x[i];
    }
}

void velocityZero( double *x, double *g, void * ){
    for (unsigned int i =0; i<nQdot; ++i) {
        g[i] =  x[nQ+i];
    }
}

void activationsZero( double *x, double *g, void * ){
    for (unsigned int i =0; i<nMus; ++i) {
        g[i] =  x[i+nQ+nQdot];
    }
}

void torquesZero( double *x, double *g, void * ){
    for (unsigned int i =0; i<nTau; ++i) {
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
    int numTag = ((int*) user_data)[0];
    s2mGenCoord Q(nQ);
    for (unsigned int i = 0; i<nQ; ++i){
        Q[i] = x[i];
    }
    s2mNodeBone tag(m.Tags(m, Q, numTag, true, true));
//    std::cout << tag.name() <<std::endl;
//    std::cout << tag << std::endl;
    g[0]=tag[0];
    g[1]=tag[1];
    g[2]=tag[2];
}

void forceConstraint( double *x, double *g, void *user_data){
//    RigidBodyDynamics::ConstraintSet CS = m.getConstraints();
//    forwardDynamicsFromMuscleActivationAndTorqueContact(x, g, user_data);
    s2mGenCoord Q(static_cast<unsigned int>(nQ));           // states
    s2mGenCoord Qdot(static_cast<unsigned int>(nQdot));     // derivated states

    // Dispatch the inputs
    for (unsigned int i = 0; i<nQ; ++i){
        Q[i] = x[i];
        Qdot[i] = x[i+nQ];
    }
    m.updateMuscles(m, Q, Qdot, true);

    std::vector<s2mMuscleStateActual> state; // controls
    for (unsigned int i = 0; i<nMus; ++i){
        state.push_back(s2mMuscleStateActual(0, x[i+nQ+nQdot]));
    }

    // Compute the torques from muscles
    s2mTau Tau = m.muscularJointTorque(m, state, false, &Q, &Qdot);
    for (unsigned int i=0; i<nTau; ++i){
        Tau[i]=Tau[i]+x[i+nQ+nQdot+nMus];
        //std::cout<<"Torques additionnels:"<<x[i+nQ+nQdot+nMus]<<std::endl;
    }
    // Compute the forward dynamics
    s2mGenCoord Qddot(nQdot);
    RigidBodyDynamics::ConstraintSet CS = m.getConstraints(m);
    RigidBodyDynamics::ForwardDynamicsContactsKokkevis(m, Q, Qdot, Tau, CS, Qddot);
    g[0]=CS.force(0);
    g[1]=CS.force(1);

}
