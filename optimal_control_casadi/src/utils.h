#ifndef UTILS_CASADI_H
#define UTILS_CASADI_H

#include "casadi/casadi.hpp"
#include "BiorbdModel.h"
extern biorbd::Model m;

namespace biorbd {
namespace utils {
class Vector;
}
}

struct ProblemSize{
    unsigned int ns; // number of shooting
    double tf; // Final time of the optimization
    double dt; // Time between two shooting

    unsigned int nu; // Number of controls
    unsigned int nx; // Number of states
};

struct BoundaryConditions{
    BoundaryConditions(
            std::vector<double> min = {},
            std::vector<double> max = {},
            std::vector<double> starting_min = {},
            std::vector<double> starting_max = {},
            std::vector<double> end_min = {},
            std::vector<double> end_max = {}
            ){
        // Dispatch intermediate states
        this->min = min;
        this->max = max;

        // If some are not declared put intermediate values instead
        starting_min.size() == 0
                ? this->starting_min = min
                : this->starting_min = starting_min;
        starting_max.size() == 0
                ? this->starting_max = max
                : this->starting_max = starting_max;
        end_min.size() == 0
                ? this->end_min = min
                : this->end_min = end_min;
        end_max.size() == 0
                ? this->end_max = max
                : this->end_max = end_max;
    }

    std::vector<double> min;
    std::vector<double> max;
    std::vector<double> starting_min;
    std::vector<double> starting_max;
    std::vector<double> end_min;
    std::vector<double> end_max;
};

struct InitialConditions{
    InitialConditions(std::vector<double> initialGuess = {}){
        this->val = initialGuess;
    }
    std::vector<double> val;
};

enum ODE_SOLVER{
    COLLOCATION,
    RK,
    CVODES
};

void defineDifferentialVariables(
        ProblemSize &ps,
        casadi::MX &u,
        casadi::MX &x);

void defineMultipleShootingNodes(
        const ProblemSize& ps,
        const BoundaryConditions &uBounds,
        const BoundaryConditions &xBounds,
        const InitialConditions &uInit,
        const InitialConditions &xInit,
        casadi::MX &V,
        BoundaryConditions &vBounds,
        InitialConditions &vInit,
        std::vector<casadi::MX> &U,
        std::vector<casadi::MX> &X);

void continuityConstraints(
        const casadi::Function& dynamics,
        const ProblemSize& ps,
        const std::vector<casadi::MX> &U,
        const std::vector<casadi::MX> &X,
        std::vector<casadi::MX> &g);

void minimizeControls(
        const ProblemSize& ps,
        const std::vector<casadi::MX> &X,
        const std::vector<casadi::MX> &U,
        casadi::MX &obj);

void solveProblemWithIpopt(
        const casadi::MX &V,
        const BoundaryConditions &vBounds,
        const InitialConditions &vInit,
        const casadi::MX &obj,
        const std::vector<casadi::MX> &constraints,
        std::vector<double>& V_opt);

void extractSolution(
        const std::vector<double>& V_opt,
        const ProblemSize& ps,
        std::vector<biorbd::utils::Vector>& Q,
        std::vector<biorbd::utils::Vector>& Qdot,
        std::vector<biorbd::utils::Vector>& Tau);

#endif
