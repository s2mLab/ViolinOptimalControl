#include "dynamics.h"
#include <rbdl/Dynamics.h>
#include "BiorbdModel.h"
#include "RigidBody/GeneralizedCoordinates.h"
#include "RigidBody/GeneralizedTorque.h"
#include "Muscles/StateDynamics.h"
#include "utils.h"


//#define CHECK_MAX_FORCE
//#define CHECK_FORCE_IF_LOW_ACTIVATION
//#define CHECK_MUSCLE_LENGTH_IS_POSITIVE

void forwardDynamics(const GeneralizedCoordinates& Q, const GeneralizedCoordinates& Qdot, const GeneralizedTorque& Tau, double *rhs){
    RigidBodyDynamics::ForwardDynamics(m, Q, Qdot, Tau, Qddot);

    for(unsigned int i = 0; i<nQ; ++i){ // Assuming nQ == nQdot
        rhs[i] = Qdot[i];
        rhs[i + nQdot] = Qddot[i];
    }
}

void forwardDynamicsFromJointTorque( double *x, double *rhs, void *){
    // Dispatch the inputs
    Dispatch_Q_Qdot(x);


    for(unsigned int i = 0; i<nTau; ++i)
        Tau[i] = x[i+nQ+nQdot+nMus];

    // Compute the forward dynamics
    forwardDynamics(Q, Qdot, Tau, rhs);
}


void forwardDynamicsFromMuscleActivation( double *x, double *rhs, void *){
    // Dispatch the inputs
    Dispatch_Q_Qdot(x);
    m.updateMuscles(Q, Qdot, true);

    for(unsigned int i = 0; i<nMus; ++i)
        musclesStates[i]->setActivation(x[i+nQ+nQdot]);

    // Compute the torques from muscles
    Tau = m.muscularJointTorque(musclesStates, false, &Q, &Qdot);

    // Compute the forward dynamics
    forwardDynamics(Q, Qdot, Tau, rhs);


    // Error checker
    // Check if Forces are not too high
    #ifdef CHECK_MAX_FORCE
        for(unsigned int int i=0; i<m.nbTau(); ++i){
            if (Tau[i]>1e5){
                std::vector<int> L;
                for(unsigned int int j=0; j<nMus; ++j){
                    if (m.musclesLengthJacobian(m, true, &Q).coeff(j,i)!=0){
                        L.push_back(j);
                    }
                }
             std::cout << "Torque "<<i<<" is too high" << std::endl;
            // std::cout << "Check the optimal lenth, the maximal force or the tendon slack lenth of muscles " << L << std::endl;
             L.clear();
             }
        }
    #endif

    // Check if muscle forces are not too high if muscle activation is low
    #ifdef CHECK_FORCE_IF_LOW_ACTIVATION
        for(unsigned int int i=0; i<m.nbTau(); ++i){
            if (Tau[i]>0.1){
                int c=0;
                std::vector<int> L;
                for(unsigned int int j=0; j<nMus; ++j){
                    if (m.musclesLengthJacobian(m, true, &Q).coeff(j,i)!=0){
                        L.push_back(j);
                        if(musclesStates[j].activation()>0.015)
                            c++;
                    }
                }
                if (c=0)
                    std::cout << "Passive force of the muscles " <<L<< " is too high. Check the tendon slack lenth."<<std::endl;
             }
        }
    #endif

    #ifdef CHECK_MUSCLE_LENGTH_IS_POSITIVE
        for(unsigned int int i=0; i<m.nbMuscleGroups(); ++i){
            for(unsigned int int j=0; j<m.muscleGroup(i).nbMuscles(); ++j)
                if (m.muscleGroup(i).muscle(j).get()->length(m, Q) <= 0)
                    std::cout << "La longueur du muscle " << i << " est inférieur á 0" <<std::endl;
        }
    #endif
}


void forwardDynamicsFromMuscleActivationAndTorque( double *x, double *rhs, void *user_data){

    Dispatch_Q_Qdot(x);
    m.updateMuscles(Q, Qdot, true);

    for(unsigned int i = 0; i<nMus; ++i){
        musclesStates[i]->setActivation(x[nQ+nQdot + i]);
    }
    // Compute the torques from muscles
    if (nMus > 0)
        Tau = m.muscularJointTorque(musclesStates, false, &Q, &Qdot);
    else
        Tau.setZero();

    for(unsigned int i=0; i<nTau; ++i){
        Tau[i] += x[nQ+nQdot+nMus + i];
    }

    // Compute the forward dynamics
    forwardDynamics(Q, Qdot, Tau, rhs);
}

void forwardDynamicsFromMuscleActivationAndTorqueContact( double *x, double *rhs, void *){
    // Dispatch the inputs
    Dispatch_Q_Qdot(x);
    m.updateMuscles(Q, Qdot, true);

    for(unsigned int i = 0; i<nMus; ++i){
        musclesStates[i]->setActivation(x[i+nQ+nQdot]);
    }

    // Compute the torques from muscles
    Tau = m.muscularJointTorque(musclesStates, false, &Q, &Qdot);
    for(unsigned int i=0; i<nTau; ++i){
        Tau[i]=Tau[i]+x[i+nQ+nQdot+nMus];
    }
    // Compute the forward dynamics
    RigidBodyDynamics::ConstraintSet& CS = m.getConstraints();
    RigidBodyDynamics::ForwardDynamicsConstraintsDirect(m, Q, Qdot, Tau, CS, Qddot);
    //RigidBodyDynamics::ForwardDynamicsContactsKokkevis(m, Q, Qdot, Tau, CS, Qddot);
    for(unsigned int i = 0; i<nQ; ++i){ // Assuming nQ == nQdot
        rhs[i] = Qdot[i];
        rhs[i + nQdot] = Qddot[i];
    }

}

void forwardDynamicsFromTorqueContact( double *x, double *rhs, void *){
    // Dispatch the inputs
    Dispatch_Q_Qdot_Tau(x);

    // Compute the forward dynamics
    RigidBodyDynamics::ConstraintSet& CS = m.getConstraints();
    RigidBodyDynamics::ForwardDynamicsConstraintsDirect(m, Q, Qdot, Tau, CS, Qddot);
    //RigidBodyDynamics::ForwardDynamicsContactsKokkevis(m, Q, Qdot, Tau, CS, Qddot);

    for(unsigned int i = 0; i<nQ; ++i){ // Assuming nQ == nQdot
        rhs[i] = Qdot[i];
        rhs[i + nQdot] = Qddot[i];
    }

}
