// C++ (and CasADi) from here on
#include <casadi.hpp>

#include "biorbdCasadi_interface_common.h"
#include "utils.h"
#include "AnimationCallback.h"

#include "biorbd.h"
extern biorbd::Model m;
biorbd::Model m("../../models/eocar.bioMod");

const std::string optimizationName("eocarBiorbd");
const std::string resultsPath("../../Results/");
const biorbd::utils::Path controlResultsFileName(resultsPath + "Controls" + optimizationName + ".txt");
const biorbd::utils::Path stateResultsFileName(resultsPath + "States" + optimizationName + ".txt");


int main(int argc, char *argv[]){
    // ---- OPTIONS ---- //
    // Dimensions of the problem
    std::cout << "Preparing the optimal control problem..." << std::endl;

    Visualization visu(Visualization::LEVEL::GRAPH, argc, argv);

    ProblemSize probSize;
    probSize.tf = 2.0;
    probSize.ns = 30;
    probSize.dt = probSize.tf/probSize.ns; // length of a control interval

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
        xBounds.starting_min.push_back(0);
        xBounds.min.push_back(-100);
        xBounds.end_min.push_back(0);

        xBounds.starting_max.push_back(0);
        xBounds.max.push_back(100);
        xBounds.end_max.push_back(0);

        xInit.val.push_back(0);
    };

    // Bounds and initial guess for the control
    BoundaryConditions uBounds;
    InitialConditions uInit;
    for (unsigned int i=0; i<m.nbMuscleTotal(); ++i) {
        uBounds.min.push_back(0);
        uBounds.max.push_back(1);
        uInit.val.push_back(0.5);
    };
    for (unsigned int i=0; i<m.nbGeneralizedTorque(); ++i) {
        uBounds.min.push_back(-100);
        uBounds.max.push_back(100);
        uInit.val.push_back(0);
    };

    // Start at the starting point and finish at the ending point
    std::vector<IndexPairing> markersToPair;
    markersToPair.push_back(IndexPairing(Instant::START, {0, 1}));
    markersToPair.push_back(IndexPairing(Instant::END, {0, 2}));

    // Always point towards the point(3)
    std::vector<IndexPairing> markerToProject;
//    markerToProject.push_back(IndexPairing (Instant::ALL, {0, 3, PLANE::XZ}));

    // Always point upward
    std::vector<IndexPairing> axesToAlign;
//    axesToAlign.push_back(IndexPairing(Instant::MID, {0, AXIS::X, 1, AXIS::MINUS_Y}));

    // Always point in line with a given vector described by markers
    std::vector<IndexPairing> alignWithMarkers;
//    alignWithMarkers.push_back(IndexPairing(Instant::MID, {0, AXIS::X, 2, 3}));

    // Always have the segment aligned with a certain system of axes
    std::vector<IndexPairing> alignWithMarkersReferenceFrame;
    alignWithMarkersReferenceFrame.push_back(IndexPairing(Instant::ALL,
            {0, AXIS::X, 1, 2, AXIS::Y, 1, 3, AXIS::X}));




    // From here, unless one wants to fundamentally change the problem,
    // they should not change anything

    // ODE right hand side
    casadi::MX states = casadi::MX::sym("x", m.nbQ()*2, 1);
    casadi::MX controls = casadi::MX::sym("p", m.nbQ(), 1);
    casadi::Function f = casadi::Function( "ForwardDyn",
                                {states, controls},
                                {ForwardDyn(m, states, controls)},
                                {"states", "controls"},
                                {"statesdot"}).expand();
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
    followMarkerConstraint(F, probSize, U, X, g, markersToPair);

    // Path constraints
    projectionOnPlaneConstraint(F, probSize, U, X, g, markerToProject);

    // Path constraints
    alignAxesConstraint(F, probSize, U, X, g, axesToAlign);

    // Path constraints
    alignAxesToMarkersConstraint(F, probSize, U, X, g, alignWithMarkers);

    // Path constraints
    alignJcsToMarkersConstraint(F, probSize, U, X, g, alignWithMarkersReferenceFrame);


    // Objective function
    casadi::MX J;
    objectiveFunction(probSize, X, U, J);

    // Online visualization
    AnimationCallback animCallback(visu, V, g, probSize, 10);

    // Optimize
    std::cout << "Solving the optimal control problem..." << std::endl;
    std::vector<double> V_opt;
    clock_t start = clock();
    solveProblemWithIpopt(V, vBounds, vInit, J, g, probSize, V_opt, animCallback);
    clock_t end=clock();
    std::cout << "Done!" << std::endl;

    // Get the optimal state trajectory
    std::vector<Eigen::VectorXd> Q;
    std::vector<Eigen::VectorXd> Qdot;
    std::vector<Eigen::VectorXd> Tau;
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
    std::vector<Eigen::VectorXd> QandQdot;
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

    while(animCallback.isActive()){}
    return 0;
}
