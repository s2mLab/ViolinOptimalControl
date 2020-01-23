// C++ (and CasADi) from here on
#include <casadi.hpp>

#include "utils.h"
#include "forward_dynamics_casadi.h"
#include "forward_kinematics_casadi.h"
#include "biorbd.h"
extern biorbd::Model m;
biorbd::Model m("../../models/BrasViolon.bioMod");

const std::string optimizationName("UpAndDowsBowCasadi");
const std::string resultsPath("../../Results/");
const biorbd::utils::Path controlResultsFileName(resultsPath + "Controls" + optimizationName + ".txt");
const biorbd::utils::Path stateResultsFileName(resultsPath + "States" + optimizationName + ".txt");


int main(){
    // ---- OPTIONS ---- //
    // Dimensions of the problem
    std::cout << "Preparing the optimal control problem..." << std::endl;
    ProblemSize probSize;
    probSize.tf = 2.0;
    probSize.ns = 30;
    probSize.dt = probSize.tf/probSize.ns; // length of a control interval

    // Functions names
    std::string dynamicsFunctionName(libforward_dynamics_casadi_name());
    std::string forwardKinFunctionName(libforward_kinematics_casadi_name());

    // Chose the ODE solver
    int odeSolver(ODE_SOLVER::RK);

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
    for (unsigned int i=0; i<m.nbGeneralizedTorque(); ++i) {
        uBounds.min.push_back(-100);
        uBounds.max.push_back(100);
        uInit.val.push_back(0);
    };

    // Bounds and initial guess for the state
    std::vector<biorbd::utils::Range> ranges;
    for (unsigned int i=0; i<m.nbSegment(); ++i){
        std::vector<biorbd::utils::Range> segRanges(m.segment(i).ranges());
        for(unsigned int j=0; j<segRanges.size(); ++j){
            ranges.push_back(segRanges[j]);
        }
    }
    BoundaryConditions xBounds;
    InitialConditions xInit;
    for (unsigned int i=0; i<m.nbQ(); ++i) {
        xBounds.starting_min.push_back(ranges[i].min());
        xBounds.min.push_back(ranges[i].min());
        xBounds.end_min.push_back(ranges[i].min());

        xBounds.starting_max.push_back(ranges[i].max());
        xBounds.max.push_back(ranges[i].max());
        xBounds.end_max.push_back(ranges[i].max());

        xInit.val.push_back(0);
    };
    for (unsigned int i=0; i<m.nbQdot(); ++i) {
        xBounds.starting_min.push_back(-100);
        xBounds.min.push_back(-100);
        xBounds.end_min.push_back(-100);

        xBounds.starting_max.push_back(100);
        xBounds.max.push_back(100);
        xBounds.end_max.push_back(100);

        xInit.val.push_back(0);
    };

    // From here, unless one wants to fundamentally change the problem,
    // they should not change anything

    // ODE right hand side
    casadi::Dict opts_dyn;
    opts_dyn["enable_fd"] = true; // This is for now, someday, it will provide the dynamic derivative!
    casadi::Function f = casadi::external(dynamicsFunctionName, opts_dyn);
    casadi::MXDict ode = {
        {"x", x},
        {"p", u},
        {"ode", f(std::vector<casadi::MX>({x, u}))[0]}
    };
    casadi::Dict ode_opt;
    ode_opt["t0"] = 0;
    ode_opt["tf"] = probSize.dt;
    if (odeSolver == ODE_SOLVER::RK || odeSolver == ODE_SOLVER::COLLOCATION)
        ode_opt["number_of_finite_elements"] = 5;
    casadi::Function F;
    if (odeSolver == ODE_SOLVER::RK)
        F = casadi::integrator("integrator", "rk", ode, ode_opt);
    else if (odeSolver == ODE_SOLVER::COLLOCATION)
        F = casadi::integrator("integrator", "collocation", ode, ode_opt);
    else if (odeSolver == ODE_SOLVER::CVODES)
        F = casadi::integrator("integrator", "cvodes", ode, ode_opt);
    else
        throw std::runtime_error("ODE solver not implemented..");

//    // Forward kinematics
//    casadi::Dict opts_forwardKin;
//    opts_forwardKin["enable_fd"] = true; // This is for now, someday, it will provide the dynamic derivative!
//    casadi::Function forwardKin = casadi::external(forwardKinFunctionName, opts_forwardKin);

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

    // Path constraints
    //pathConstraints(F, forwardKin, probSize, U, X, g);

    // Objective function
    casadi::MX J;
    objectiveFunction(probSize, X, U, J);

    // Optimize
    std::cout << "Solving the optimal control problem..." << std::endl;
    std::vector<double> V_opt;
    clock_t start = clock();
    solveProblemWithIpopt(V, vBounds, vInit, J, g, V_opt);
    clock_t end=clock();
    std::cout << "Done!" << std::endl;

    // Get the optimal state trajectory
    std::vector<biorbd::rigidbody::GeneralizedCoordinates> Q;
    std::vector<biorbd::rigidbody::GeneralizedVelocity> Qdot;
    std::vector<biorbd::rigidbody::GeneralizedTorque> Tau;
    extractSolution(V_opt, probSize, Q, Qdot, Tau);

    // Show the solution
    std::cout << "Results:" << std::endl;
    for (unsigned int q=0; q<m.nbQ(); ++q){
        std::cout << "Q[" << q <<"] = " << Q[q].transpose() << std::endl;
        std::cout << "Qdot[" << q <<"] = " << Qdot[q].transpose() << std::endl;
        std::cout << "Tau[" << q <<"] = " << Tau[q].transpose() << std::endl;
        std::cout << std::endl;
    }
    createTreePath(resultsPath);
    writeCasadiResults(controlResultsFileName, Tau, probSize.dt);
    std::vector<biorbd::utils::Vector> QandQdot;
    for (auto q : Q){
        QandQdot.push_back(q);
    }
    for (auto qdot : Qdot){
        QandQdot.push_back(qdot);
    }
    writeCasadiResults(controlResultsFileName, Tau, probSize.dt);
    writeCasadiResults(stateResultsFileName, QandQdot, probSize.dt);

    // ---------- FINALIZE  ------------ //
    double time_exec(double(end - start)/CLOCKS_PER_SEC);
    std::cout<<"Execution time: "<<time_exec<<std::endl;
    return 0;
}
