#include "dynamics.h"

void forwardDynamics(const s2mGenCoord& Q, const s2mGenCoord& Qdot, const s2mTau& Tau, double *rhs){
    s2mGenCoord Qddot(nQdot);
    RigidBodyDynamics::ForwardDynamics(m, Q, Qdot, Tau, Qddot);

    for (unsigned int i = 0; i<nQ; ++i){ // Assuming nQ == nQdot
        rhs[i] = Qdot[i];
        rhs[i + nQdot] = Qddot[i];
    }
}

void forwardDynamicsFromJointTorque( double *x, double *rhs, void *){
    s2mGenCoord Q(nQ);
    s2mGenCoord Qdot(nQdot);
    s2mTau Tau(nTau); // controls

    // Dispatch the inputs
    for (unsigned int i = 0; i<nQ; ++i){ // Assuming nQ == nQdot
        Q[i] = x[i];
        Qdot[i] = x[i+nQ];
    }
    for (unsigned int i = 0; i<nTau; ++i)
        Tau[i] = x[i+nQ+nQdot];

    // Compute the forward dynamics
    forwardDynamics(Q, Qdot, Tau, rhs);
}

void forwardDynamicsFromMuscleActivation( double *x, double *rhs, void *){
    s2mGenCoord Q(static_cast<unsigned int>(nQ));           // states
    s2mGenCoord Qdot(static_cast<unsigned int>(nQdot));     // derivated states

    // Dispatch the inputs
    for (unsigned int i = 0; i<nQ; ++i){
        Q[i] = x[i];
        Qdot[i] = x[i+nQ];
    }
    m.updateMuscles(m, Q, Qdot, true);

    std::vector<s2mMuscleStateActual> state;// controls
    for (unsigned int i = 0; i<nMus; ++i)
        state.push_back(s2mMuscleStateActual(0, x[i+nQ+nQdot]));

    // Compute the torques from muscles
    s2mTau Tau = m.muscularJointTorque(m, state, true, &Q, &Qdot);

    // Compute the forward dynamics
    forwardDynamics(Q, Qdot, Tau, rhs);


    // Error checker
    // Check if Forces are not too high
    #ifdef CHECK_MAX_FORCE
        for(int i=0; i<m.nbTau(); ++i){
            if (Tau[i]>1e5){
                std::vector<int> L;
                for(int j=0; j<nMus; ++j){
                    if (m.musclesLengthJacobian(m, true, &Q).coeff(j,i)!=0){
                        L.push_back(j);
                    }
                }
             std::cout << "Torque "<<i<<" is too high" << endl;
             std::cout << "Check the optimal lenth, the maximal force or the tendon slack lenth of muscles " << L <<endl;
             L.clear();
             }
        }
    #endif

    // Check if muscle forces are not too high if muscle activation is low
    #ifdef CHECK_FORCE_IF_LOW_ACTIVATION
        for(int i=0; i<m.nbTau(); ++i){
            if (Tau[i]>0.1){
                int c=0;
                std::vector<int> L;
                for(int j=0; j<nMus; ++j){
                    if (m.musclesLengthJacobian(m, true, &Q).coeff(j,i)!=0){
                        L.push_back(j);
                        if(state[j].activation()>0.015)
                            c++;
                    }
                }
                if (c=0)
                    std::cout << "Passive force of the muscles " <<L<< " is too high. Check the tendon slack lenth."<<endl;
             }
        }
    #endif

    #ifdef CHECK_MUSCLE_LENGTH_IS_POSITIVE
        for(int i=0; i<m.nbMuscleGroups(); ++i){
            for(int j=0; j<m.muscleGroup(i).nbMuscles(); ++j)
                if (m.muscleGroup(i).muscle(j).get()->length(m, Q) <= 0)
                    std::cout << "La longueur du muscle " << i << " est inférieur á 0" <<endl;
        }
    #endif
}
