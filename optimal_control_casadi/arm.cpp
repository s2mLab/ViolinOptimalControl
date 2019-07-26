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
    BoundaryConditions uBounds = {
        { 0 }, // initial guess
        { -100 }, // min
        { 100 }, // max
    };

    // Bounds and initial guess for the state
    BoundaryConditions xBounds = {
        { 0, 0 }, // initial guess
        { -500, -500 }, // min
        { 500, 500 }, // max
        { 0, 0 }, // starting_min
        { 0, 0 }, // starting_max
        { 100, 0 }, // end_min
        { 100, 0 } // end_max
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
    std::vector<casadi::MX> U, X;
    defineMultipleShootingNodes(probSize, uBounds, xBounds, V, vBounds, U, X);

    // Continuity constraints
    std::vector<casadi::MX> g;
    continuityConstraints(F, probSize, U, X, g);

    // Objective function
    casadi::MX J;
    objectiveFunction(probSize, X, U, J);

    // Optimize
    std::vector<double> V_opt;
    solveProblemWithIpopt(V, vBounds, J, g, V_opt);

    // Get the optimal state trajectory
    s2mVector Q;
    s2mVector Qdot;
    s2mVector Tau;
    extractSolution(V_opt, probSize, Q, Qdot, Tau);

    // Show the solution
    std::cout << "Q = " << Q.transpose() << std::endl;
    std::cout << "Qdot = " << Qdot.transpose() << std::endl;
    std::cout << "Tau = " << Tau.transpose() << std::endl;

    return 0;
}
