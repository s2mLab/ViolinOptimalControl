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

void fillSparsity(){
    if (!isSparsityFilled){
        assert(m.nbQ() == m.nbQdot()); // Quaternions are not implemented so far
        assert(m.nbQdot() == m.nbGeneralizedTorque()); // Free falling model

        Q_sparsity[0] = m.nbQ();
        Qdot_sparsity[0] = m.nbQdot();
        Qddot_sparsity[0] = m.nbQdot();
        Tau_sparsity[0] = m.nbGeneralizedTorque();
        Xp_sparsity[0] = m.nbQ() + m.nbQdot();
        isSparsityFilled = true;
    }
}

const char* libforward_dynamics_casadi_name(void){
    return "libforward_dynamics_casadi";
}

int libforward_dynamics_casadi(const casadi_real** arg, casadi_real** res, casadi_int*, casadi_real*, void*){
    biorbd::rigidbody::GeneralizedCoordinates Q(m), Qdot(m), Qddot(m);
    biorbd::rigidbody::GeneralizedTorque Tau(m);

    // Dispatch data
    for (unsigned int i = 0; i < m.nbQ(); ++i){
        Q[i] = arg[0][i];
        Qdot[i] = arg[0][i+m.nbQ()];
        Tau[i] = arg[1][i];
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
    fillSparsity();
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
    fillSparsity();
    switch (i) {
        case 0: return Xp_sparsity;
    default: return nullptr;
    }
}

int libforward_dynamics_casadi_work(casadi_int *sz_arg,
                                               casadi_int* sz_res,
                                               casadi_int *sz_iw,
                                               casadi_int *sz_w) {
    if (sz_arg) *sz_arg = m.nbQ() + m.nbQdot() + m.nbGeneralizedTorque();
    if (sz_res) *sz_res = m.nbQdot() + m.nbQddot();
    if (sz_iw) *sz_iw = 0;
    if (sz_w) *sz_w = 0;
    return 0;
}

#ifdef __cplusplus
} /* extern "C" */
#endif
