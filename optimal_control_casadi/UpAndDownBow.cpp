// C++ (and CasADi) from here on
#include <casadi.hpp>

#include "utils.h"
#include "forward_dynamics_casadi.h"
#include "forward_kinematics_casadi.h"
#include "projectionOnSegment_casadi.h"
#include "angle_between_segments_casadi.h"
#include "angle_between_segment_and_markers_casadi.h"

#include "biorbd.h"
extern biorbd::Model m;

biorbd::Model m("../../models/BrasViolon.bioMod");
const std::string optimizationName("UpAndDowsBowCasadi");
const ViolinStringNames stringPlayed(ViolinStringNames::E);

static unsigned int idxSegmentBow(8);
static unsigned int idxSegmentViolin(16);
static unsigned int tagBowFrog(16);
static unsigned int tagBowTip(18);
static unsigned int tagViolinBStringBridge(42);
static unsigned int tagViolinEStringBridge(34);
static unsigned int tagViolinEStringNeck(35);
static unsigned int tagViolinAStringBridge(36);
static unsigned int tagViolinAStringNeck(37);
static unsigned int tagViolinDStringBridge(38);
static unsigned int tagViolinDStringNeck(39);
static unsigned int tagViolinGStringBridge(40);
static unsigned int tagViolinGStringNeck(41);
static unsigned int tagViolinCStringBridge(43);

// The following values for initialization were determined using the "find_initial_pose.py" script
const std::vector<double> initQFrogOnGString =
{-0.07018473, -0.0598567, 1.1212999, 0.90238053, 1.4856272, -0.09812186, 0.1498479, -0.48374356, -0.41007239};
const std::vector<double> initQTipOnGString =
{0.07919993, -0.80789739, 0.96181894, 0.649565, 0.24537634, -0.16297839, 0.0659226, 0.14617512, 0.49722962};
const std::vector<double> initQFrogOnDString =
{0.02328389, 0.03568661, 0.99308077, 0.93208313, 1.48380368, -0.05939737, 0.26778731, -0.50387155, -0.43094647};
const std::vector<double> initQTipOnDString =
{0.08445998, -0.66837886, 0.91480044, 0.66766976, 0.25110097, -0.07526545, 0.09689908, -0.00561614, 0.62755419};
const std::vector<double> initQFrogOnAString =
{-0.0853157, -0.03099135, 1.04751851, 0.93222374, 1.50707542, -0.12888636, 0.04174079, -0.57032577, -0.31175627};
const std::vector<double> initQTipOnAString =
{-0.06506266, -0.44904332, 0.97090727, 1.00055219, 0.17983593, -0.3363436, -0.03281452, 0.04678383, 0.64465767};
const std::vector<double> initQFrogOnEString =
{-0.19923285, 0.08890963, 0.99469991, 0.97362544, 1.48863482, -0.09960671, 0.01607784, -0.44009434, -0.36712403};
const std::vector<double> initQTipOnEString =
{0.03328374, -0.27888401, 0.7623438, 0.59379268, 0.16563931, 0.2443971, 0.1824652, 0.1587049, 0.52812319};

const std::string resultsPath("../../Results/");
const biorbd::utils::Path controlResultsFileName(resultsPath + "Controls" + optimizationName + ".txt");
const biorbd::utils::Path stateResultsFileName(resultsPath + "States" + optimizationName + ".txt");


int main(){
    // ---- OPTIONS ---- //
    // Dimensions of the problem
    std::cout << "Preparing the optimal control problem..." << std::endl;
    ProblemSize probSize;
    probSize.tf = 0.5;
    probSize.ns = 30;
    probSize.dt = probSize.tf/probSize.ns; // length of a control interval

    // Functions names
    std::string dynamicsFunctionName(libforward_dynamics_casadi_name());
    std::string forwardKinFunctionName(libforward_kinematics_casadi_name());
    std::string projectionFunctionName(libprojectionOnSegment_casadi_name());
    std::string axesFunctionName(libangle_between_segments_casadi_name());
    std::string axesToMarkersFunctionName(libangle_between_segment_and_markers_casadi_name());

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
    std::vector<double> initQFrog;
    std::vector<double> initQTip;
    if (stringPlayed == ViolinStringNames::E){
        initQFrog = initQFrogOnEString;
        initQTip = initQTipOnEString;
    }
    else if (stringPlayed == ViolinStringNames::A){
        initQFrog = initQFrogOnAString;
        initQTip = initQTipOnAString;
    }
    else if (stringPlayed == ViolinStringNames::D){
        initQFrog = initQFrogOnDString;
        initQTip = initQTipOnDString;
    }
    else if (stringPlayed == ViolinStringNames::G){
        initQFrog = initQFrogOnGString;
        initQTip = initQTipOnGString;
    }
    biorbd::rigidbody::GeneralizedCoordinates initQ(m);
    biorbd::rigidbody::GeneralizedVelocity initQdot(m);
    biorbd::rigidbody::GeneralizedAcceleration initQddot(m);
    for (unsigned int i=0; i<m.nbQ(); ++i){
        initQ(i) = initQFrog[i];
        initQdot(i) = 0;
        initQddot(i) = 0;
    }
    m.UpdateKinematicsCustom(&initQ, &initQdot, &initQddot);
    biorbd::rigidbody::GeneralizedTorque initTau(m);
    RigidBodyDynamics::InverseDynamics(m, initQ, initQdot, initQddot, initTau);
    for (unsigned int i=0; i<m.nbQ(); ++i) {
        xBounds.starting_min.push_back(ranges[i].min());
        xBounds.min.push_back(ranges[i].min());
        xBounds.end_min.push_back(ranges[i].min());

        xBounds.starting_max.push_back(ranges[i].max());
        xBounds.max.push_back(ranges[i].max());
        xBounds.end_max.push_back(ranges[i].max());

        xInit.val.push_back(initQFrog[i]);
    };
    for (unsigned int i=0; i<m.nbQdot(); ++i) {
        xBounds.starting_min.push_back(-100);
        xBounds.min.push_back(-100);
        xBounds.end_min.push_back(-100);

        xBounds.starting_max.push_back(100);
        xBounds.max.push_back(100);
        xBounds.end_max.push_back(100);

        xInit.val.push_back(initTau(i));
    };

    // Get to frog a`nd tip at beggining and end
    unsigned int stringIdx;
    if (stringPlayed == ViolinStringNames::E){
        stringIdx = tagViolinEStringBridge;
    }
    else if (stringPlayed == ViolinStringNames::A){
        stringIdx = tagViolinAStringBridge;
    }
    else if (stringPlayed == ViolinStringNames::D){
        stringIdx = tagViolinDStringBridge;
    }
    else if (stringPlayed == ViolinStringNames::G){
        stringIdx = tagViolinGStringBridge;
    }
    std::vector<IndexPairing> markersToPair;
    markersToPair.push_back(IndexPairing(Instant::START, {tagBowFrog, stringIdx}));
    markersToPair.push_back(IndexPairing(Instant::END, {tagBowTip, stringIdx}));

    // Keep the bow on the string
    std::vector<IndexPairing> markerToProject;
    markerToProject.push_back(
                IndexPairing (Instant::ALL, {idxSegmentBow, stringIdx, PLANE::XZ}));

    // Have the bow to lie on the string
    unsigned int idxLowStringBound;
    unsigned int idxHighStringBound;
    if (stringPlayed == ViolinStringNames::E){
        idxLowStringBound = tagViolinAStringBridge;
        idxHighStringBound = tagViolinBStringBridge;
    }
    else if (stringPlayed == ViolinStringNames::A){
        idxLowStringBound = tagViolinDStringBridge;
        idxHighStringBound = tagViolinEStringBridge;
    }
    else if (stringPlayed == ViolinStringNames::D){
        idxLowStringBound = tagViolinGStringBridge;
        idxHighStringBound = tagViolinAStringBridge;
    }
    else if (stringPlayed == ViolinStringNames::G){
        idxLowStringBound = tagViolinCStringBridge;
        idxHighStringBound = tagViolinDStringBridge;
    }
    std::vector<IndexPairing> alignWithMarkers;
    alignWithMarkers.push_back(IndexPairing(
                    Instant::MID,
                    {idxSegmentBow, AXIS::MINUS_Y, idxHighStringBound, idxLowStringBound}));

    // No need to aligning two segments
    std::vector<IndexPairing> axesToAlign;



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

    // Forward kinematics
    casadi::Dict opts_forwardKin;
    opts_forwardKin["enable_fd"] = true; // This is for now, someday, it will provide the dynamic derivative!
    casadi::Function forwardKin = casadi::external(forwardKinFunctionName, opts_forwardKin);

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
    followMarkerConstraint(F, forwardKin, probSize, U, X, g, markersToPair);

    // Path constraints
    casadi::Dict opts_projectionFunction;
    opts_projectionFunction["enable_fd"] = true;
    casadi::Function projectionFunction = casadi::external(projectionFunctionName, opts_projectionFunction);
    projectionOnPlaneConstraint(F, projectionFunction, probSize, U, X, g, markerToProject);

    // Path constraints
    casadi::Dict opts_axesFunction;
    opts_axesFunction["enable_fd"] = true;
    casadi::Function axesFunction = casadi::external(axesFunctionName, opts_axesFunction);
    alignAxesConstraint(F, axesFunction, probSize, U, X, g, axesToAlign);

    // Path constraints
    casadi::Dict opts_axesToMarkersFunction;
    opts_axesToMarkersFunction["enable_fd"] = true;
    casadi::Function axesToMarkersFunction = casadi::external(axesToMarkersFunctionName, opts_axesToMarkersFunction);
    alignAxesToMarkersConstraint(F, axesToMarkersFunction, probSize, U, X, g, alignWithMarkers);

    // Objective function
    casadi::MX J;
    objectiveFunction(probSize, X, U, J);

    // Optimize
    std::cout << "Solving the optimal control problem..." << std::endl;
    clock_t start = clock();
    std::vector<double> V_opt;
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
