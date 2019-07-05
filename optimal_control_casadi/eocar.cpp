// C++ (and CasADi) from here on
#include <casadi/casadi.hpp>
#include "s2mMusculoSkeletalModel.h"

#include "include/forward_dynamics_casadi.h"
s2mMusculoSkeletalModel m("../../models/simple.bioMod");

int main(){
    // Use CasADi's "external" to load the compiled function
    casadi::Function f = casadi::external(libforward_dynamics_casadi_name());

    casadi::DM Q(reshape(casadi::DM(std::vector<double>({0, 0, 0, 0, 0, 0})), m.nbQ(), 1));
    casadi::DM Qdot(reshape(casadi::DM(std::vector<double>({0, 0, 0, 0, 0, 0})), m.nbQdot(), 1));
    casadi::DM Tau(reshape(casadi::DM(std::vector<double>({0, 0, 0, 0, 0, 0})), m.nbTau(), 1));

    // Use like any other CasADi function
    std::vector<casadi::DM> arg = {Q, Qdot, Tau};
    std::vector<casadi::DM> res = f(arg);

    std::cout << "Qdot: " << res.at(0) << std::endl;
    std::cout << "Qddot: " << res.at(1) << std::endl;

    return 0;
}
