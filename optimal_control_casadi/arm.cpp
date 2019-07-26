// C++ (and CasADi) from here on
#include <casadi/casadi.hpp>
#include "casadi/core/optistack.hpp"

#include "utils.h"
#include "forward_dynamics_casadi.h"
#include "s2mMusculoSkeletalModel.h"
extern s2mMusculoSkeletalModel m;
s2mMusculoSkeletalModel m("../../models/simple.bioMod");

int main(){
    // Dimensions of the problem
    ProblemSize probSize;
    probSize.tf = 5.0;
    probSize.ns = 30;
    probSize.dt = probSize.tf/probSize.ns; // length of a control interval

    // Functions names
    std::string dynamicsFunctionName(libforward_dynamics_casadi_name());

    // Chose the objective function
    void (*objectiveFunction)(
                const ProblemSize&,
                const std::vector<casadi::MX>&,
                const std::vector<casadi::MX>&,
                casadi::MX&) = minimizeControls;

    // Differential variables
    casadi::MX u;
    casadi::MX x;
    defineDifferentialVariables(probSize, u, x);

    // Bounds and initial guess for the control
    BoundaryConditions uBounds;
    InitialConditions uInit;
    for (unsigned int i=0; i<m.nbTau(); ++i) {
        uBounds.min.push_back(-100);
        uBounds.max.push_back(100);
        uInit.val.push_back(0);
    };

    // Bounds and initial guess for the state
    BoundaryConditions xBounds;
    InitialConditions xInit;
    for (unsigned int i=0; i<m.nbQ(); ++i) {
        xBounds.min.push_back(-500);
        xBounds.starting_min.push_back(0);
        if (i == 0) xBounds.end_min.push_back(100);
        else if (i == 1) xBounds.end_min.push_back(50);

        xBounds.max.push_back(500);
        xBounds.starting_max.push_back(0);
        if (i == 0) xBounds.end_max.push_back(100);
        else if (i == 1) xBounds.end_max.push_back(50);

        xInit.val.push_back(0);
    };
    for (unsigned int i=0; i<m.nbQdot(); ++i) {
        xBounds.min.push_back(-500);
        xBounds.starting_min.push_back(0);
        xBounds.end_min.push_back(0);

        xBounds.max.push_back(500);
        xBounds.starting_max.push_back(0);
        xBounds.end_max.push_back(0);

        xInit.val.push_back(0);
    };



    // From here, unless one wants to fundamentally change the problem,
    // they should not change anything

    // ODE right hand side
    casadi::Dict opts_dyn;
    opts_dyn["enable_fd"] = true; // This is for now, someday, it will provide the dynamic derivative!
    casadi::Function f = casadi::external(dynamicsFunctionName, opts_dyn);
    casadi::MXDict ode = {{"x", x}, {"p", u}, {"ode", f(std::vector<casadi::MX>({x, u}))[0]}};
    casadi::Function F( casadi::integrator("integrator", "cvodes", ode, {{"t0", 0}, {"tf", probSize.dt}} ) );

    // Prepare the NLP problem
    casadi::MX V;
    BoundaryConditions vBounds;
    InitialConditions vInit;
    std::vector<casadi::MX> U;
    std::vector<casadi::MX> X;
    defineMultipleShootingNodes(probSize, uBounds, xBounds, uInit, xInit,
                                V, vBounds, vInit, U, X);

    // Continuity constraints
    std::vector<casadi::MX> g;
    continuityConstraints(F, probSize, U, X, g);

    // Objective function
    casadi::MX J;
    objectiveFunction(probSize, X, U, J);

    // Optimize
    std::vector<double> V_opt;
    solveProblemWithIpopt(V, vBounds, vInit, J, g, V_opt);

    // Get the optimal state trajectory
    std::vector<s2mVector> Q;
    std::vector<s2mVector> Qdot;
    std::vector<s2mVector> Tau;
    extractSolution(V_opt, probSize, Q, Qdot, Tau);

    // Show the solution
    for (unsigned int q=0; q<m.nbQ(); ++q){
        std::cout << "Q[" << q <<"] = " << Q[q].transpose() << std::endl;
        std::cout << "Qdot[" << q <<"] = " << Qdot[q].transpose() << std::endl;
        std::cout << "Tau[" << q <<"] = " << Tau[q].transpose() << std::endl;
        std::cout << std::endl;
    }
    return 0;
}
