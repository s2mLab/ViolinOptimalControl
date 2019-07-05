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

void fillSparsity(){
    if (!isSparsityFilled){
        Q_sparsity[0] = m.nbQ();
        Qdot_sparsity[0] = m.nbQdot();
        Qddot_sparsity[0] = m.nbQddot();
        Tau_sparsity[0] = m.nbTau();
        isSparsityFilled = true;
    }
}

const char* libforward_dynamics_casadi_name(void){
    return "libforward_dynamics_casadi";
}

int libforward_dynamics_casadi(const casadi_real** arg, casadi_real** res, casadi_int*, casadi_real*, void*){
    s2mGenCoord Q(m), Qdot(m), Qddot(m);
    s2mTau Tau(m);

    // Dispatch data
    for (unsigned int i = 0; i < m.nbQ(); ++i){
        Q[i] = arg[0][i];
        Qdot[i] = arg[1][i];
        Tau[i] = arg[2][i];
    }

    // Perform the forward dynamics
    RigidBodyDynamics::ForwardDynamics(m, s2mGenCoord(m).setZero(), s2mGenCoord(m).setZero(), Tau, Qddot);

    // Return the answers
    for (unsigned int i = 0; i< m.nbQ(); ++i){
        res[0][i] = Qdot[i];
        res[1][i] = Qddot[i];
    }

    // DEBUG
    std::cout << "Q = " << Q.transpose() << std::endl;
    std::cout << "Qdot = " << Qdot.transpose() << std::endl;
    std::cout << "Tau = " << Tau.transpose() << std::endl;
    std::cout << "Qddot = " << Qddot.transpose() << std::endl;
    return 0;
}

// IN
casadi_int libforward_dynamics_casadi_n_in(void){
    return 3;
}
const char* libforward_dynamics_casadi_name_in(casadi_int i){
    switch (i) {
    case 0: return "Q";
    case 1: return "Qdot";
    case 2: return "Tau";
    default: return nullptr;
    }
}
const casadi_int* libforward_dynamics_casadi_sparsity_in(casadi_int i) {
    fillSparsity();
    switch (i) {
    case 0: return Q_sparsity;
    case 1: return Qdot_sparsity;
    case 2: return Tau_sparsity;
    default: return nullptr;
    }
}

// OUT
casadi_int libforward_dynamics_casadi_n_out(void){
    return 2;
}
const char* libforward_dynamics_casadi_name_out(casadi_int i){
    switch (i) {
    case 0: return "Qdot";
    case 1: return "Qddot";
    default: return nullptr;
    }
}
const casadi_int* libforward_dynamics_casadi_sparsity_out(casadi_int i) {
    fillSparsity();
    switch (i) {
    case 0: return Qdot_sparsity;
    case 1: return Qddot_sparsity;
    default: return nullptr;
    }
}

int libforward_dynamics_casadi_work(casadi_int *sz_arg,
                                               casadi_int* sz_res,
                                               casadi_int *sz_iw,
                                               casadi_int *sz_w) {
    if (sz_arg) *sz_arg = m.nbQ() + m.nbQdot() + m.nbTau();
    if (sz_res) *sz_res = m.nbQdot() + m.nbQddot();
    if (sz_iw) *sz_iw = 0;
    if (sz_w) *sz_w = 0;
    return 0;
}

#ifdef __cplusplus
} /* extern "C" */
#endif
