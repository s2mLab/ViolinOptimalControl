#include "forward_dynamics_casadi.h"

#ifdef __cplusplus
extern "C" {
#endif

// Declare sparsity dimensions
static bool isSparsityFilled(false);
static casadi_int Q_sparsity[3] = {-1, 1, 1};
static casadi_int Qdot_sparsity[3] = {-1, 1, 1};
static casadi_int Qddot_sparsity[3] = {-1, 1, 1};
static casadi_int Tau_sparsity[3] = {-1, 1, 1};
static casadi_int Xp_sparsity[3] = {-1, 1, 1};

static biorbd::rigidbody::GeneralizedCoordinates Q;
static biorbd::rigidbody::GeneralizedVelocity Qdot;
static biorbd::rigidbody::GeneralizedAcceleration Qddot;
biorbd::rigidbody::GeneralizedTorque Tau;
static std::vector<std::shared_ptr<biorbd::muscles::StateDynamics>> musclesStates;

void libforward_dynamics_casadi_fillSparsity(){
    if (!isSparsityFilled){
        assert(m.nbQ() == m.nbQdot()); // Quaternions are not implemented so far
        assert(m.nbQdot() == m.nbGeneralizedTorque()); // Free falling model

        Q_sparsity[0] = m.nbQ();
        Qdot_sparsity[0] = m.nbQdot();
        Qddot_sparsity[0] = m.nbQdot();
        Tau_sparsity[0] = m.nbMuscleTotal() + m.nbGeneralizedTorque();
        Xp_sparsity[0] = m.nbQ() + m.nbQdot();
        isSparsityFilled = true;


        // Allocate proper memory
        Q = biorbd::rigidbody::GeneralizedCoordinates(m);
        Qdot = biorbd::rigidbody::GeneralizedVelocity(m);
        Qddot = biorbd::rigidbody::GeneralizedAcceleration(m);
        Tau = biorbd::rigidbody::GeneralizedTorque(m);
        for(unsigned int i = 0; i<m.nbMuscleTotal(); ++i){
            musclesStates.push_back(std::make_shared<biorbd::muscles::StateDynamics>(
                        biorbd::muscles::StateDynamics()));
        }
    }
}

const char* libforward_dynamics_casadi_name(void){
    return "libforward_dynamics_casadi";
}

int libforward_dynamics_casadi(
        const casadi_real** arg,
        casadi_real** res,
        casadi_int*,
        casadi_real*,
        void*){

    // Dispatch data
    for (unsigned int i = 0; i < m.nbQ(); ++i){
        Q[i] = arg[0][i];
        Qdot[i] = arg[0][i+m.nbQ()];
    }
    for (unsigned int i=0; i<m.nbMuscleTotal(); ++i){
        musclesStates[i]->setActivation(arg[1][i]);
    }
    if (m.nbMuscleTotal() > 0){
        Tau = m.muscularJointTorque(musclesStates, true, &Q, &Qdot);
    } else {
        Tau.setZero();
    }
    for (unsigned int i=0; i<m.nbGeneralizedTorque(); ++i){
        Tau[i] += arg[1][i + m.nbMuscleTotal()];
    }

    // Perform the forward dynamics
    RigidBodyDynamics::ForwardDynamics(m, Q, Qdot, Tau, Qddot);

    // Return the answers
    for (unsigned int i = 0; i< m.nbQ(); ++i){
        res[0][i] = Qdot[i];
        res[0][i+m.nbQ()] = Qddot[i];
    }

    return 0;
}

// IN
casadi_int libforward_dynamics_casadi_n_in(void){
    return 2;
}
const char* libforward_dynamics_casadi_name_in(casadi_int i){
    switch (i) {
    case 0: return "States";
    case 1: return "Tau";
    default: return nullptr;
    }
}
const casadi_int* libforward_dynamics_casadi_sparsity_in(casadi_int i) {
    libforward_dynamics_casadi_fillSparsity();
    switch (i) {
        case 0: return Xp_sparsity;
        case 1: return Tau_sparsity;
        default: return nullptr;
    }
}

// OUT
casadi_int libforward_dynamics_casadi_n_out(void){
    return 1;
}
const char* libforward_dynamics_casadi_name_out(casadi_int i){
    switch (i) {
        case 0: return "StatesDot";
        default: return nullptr;
    }
}
const casadi_int* libforward_dynamics_casadi_sparsity_out(casadi_int i) {
    libforward_dynamics_casadi_fillSparsity();
    switch (i) {
        case 0: return Xp_sparsity;
    default: return nullptr;
    }
}

#ifdef __cplusplus
} /* extern "C" */
#endif
