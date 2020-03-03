// C++ (and CasADi) from here on
#include <casadi.hpp>

#include "utils.h"
#include "biorbdCasadi_interface_common.h"
#include "AnimationCallback.h"

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
{-0.26963739, -0.37332812,  0.55297438,  1.16757958,  1.5453081, 0.08781926,  0.66038247, -0.58420915, -0.6424003};
const std::vector<double> initQTipOnGString =
{-0.01828739, -1.31128207,  0.19282409,  0.60925735,  0.70654631, -0.07557834,  0.17204947,  0.11369929,  0.26267182};
const std::vector<double> initQFrogOnDString =
{-0.12599098, -0.45205593,  0.5822579 ,  1.11068584,  1.45957662, 0.11948427,  0.50336002, -0.40407875, -0.456703117};
const std::vector<double> initQTipOnDString =
{0.03788864, -0.70345511,  0.23451146,  0.9479002 ,  0.11111476, 0.41349365,  0.24701369,  0.2606112 ,  0.48426223};
const std::vector<double> initQFrogOnAString =
{-0.15691089, -0.52162508,  0.59001626,  1.10637291,  1.47285539, 0.03932967,  0.31431404, -0.39598565, -0.44465406};
const std::vector<double> initQTipOnAString =
{0.03051712, -0.69048243,  0.36951694,  0.88094724,  0.15574657, 0.29978535,  0.20718762,  0.14710871,  0.55469901};
const std::vector<double> initQFrogOnEString =
{-0.32244523, -0.45567388,  0.69477217,  1.14551489,  1.40942749, -0.10300415,  0.14266607, -0.23330034, -0.25421303};
const std::vector<double> initQTipOnEString =
{ 0.08773515, -0.56553214,  0.64993785,  1.0591878 , -0.18567152, 0.24296588,  0.15829188,  0.21021353,  0.71442364};



int main(int argc, char *argv[]){
    // ---- OPTIONS ---- //
    // Dimensions of the problem
    std::cout << "Preparing the optimal control problem..." << std::endl;

    Visualization visu(Visualization::LEVEL::GRAPH, argc, argv);

    ProblemSize probSize;
    probSize.tf = 0.5;
    probSize.ns = 30;
    probSize.dt = probSize.tf/probSize.ns; // length of a control interval

    // Chose the ODE solver
    ODE_SOLVER odeSolver(ODE_SOLVER::RK);

    // Chose the objective functions
    std::vector<std::pair<void (*)(const ProblemSize&,
                         const std::vector<casadi::MX>&,
                         const std::vector<casadi::MX>&,
                         double,
                         casadi::MX&), double>> objectiveFunctions;
    objectiveFunctions.push_back(std::make_pair(minimizeTorqueControls, 100));
    objectiveFunctions.push_back(std::make_pair(minimizeMuscleControls, 1));
    objectiveFunctions.push_back(std::make_pair(minimizeStates, 1.0/10));

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
    Eigen::VectorXd initTau(m.nbGeneralizedTorque());
    initTau.setZero(); // Inverse dynamics?
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
        xBounds.starting_min.push_back(0);
        xBounds.min.push_back(-100);
        xBounds.end_min.push_back(0);

        xBounds.starting_max.push_back(0);
        xBounds.max.push_back(100);
        xBounds.end_max.push_back(0);

        xInit.val.push_back(initTau(i));
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

    // Prepare constraints
    unsigned int stringBridgeIdx;
    unsigned int stringNeckIdx;
    unsigned int idxLowStringBound;
    unsigned int idxHighStringBound;
    if (stringPlayed == ViolinStringNames::E){
        stringBridgeIdx = tagViolinEStringBridge;
        stringNeckIdx = tagViolinEStringNeck;
        idxLowStringBound = tagViolinAStringBridge;
        idxHighStringBound = tagViolinBStringBridge;
    }
    else if (stringPlayed == ViolinStringNames::A){
        stringBridgeIdx = tagViolinAStringBridge;
        stringNeckIdx = tagViolinAStringNeck;
        idxLowStringBound = tagViolinDStringBridge;
        idxHighStringBound = tagViolinEStringBridge;
    }
    else if (stringPlayed == ViolinStringNames::D){
        stringBridgeIdx = tagViolinDStringBridge;
        stringNeckIdx = tagViolinDStringNeck;
        idxLowStringBound = tagViolinGStringBridge;
        idxHighStringBound = tagViolinAStringBridge;
    }
    else if (stringPlayed == ViolinStringNames::G){
        stringBridgeIdx = tagViolinGStringBridge;
        stringNeckIdx = tagViolinGStringNeck;
        idxLowStringBound = tagViolinCStringBridge;
        idxHighStringBound = tagViolinDStringBridge;
    }

    // If the movement is cyclic
    bool useCyclicObjective = false;
    bool useCyclicConstraint = false;

    // Start at frog and get tip at end
    std::vector<IndexPairing> markersToPair;
    markersToPair.push_back(IndexPairing(Instant::START, {tagBowFrog, stringBridgeIdx}));
    markersToPair.push_back(IndexPairing(Instant::MID, {tagBowTip, stringBridgeIdx}));
    markersToPair.push_back(IndexPairing(Instant::END, {tagBowFrog, stringBridgeIdx}));

    // Keep the string targetted by the bow
    std::vector<IndexPairing> markerToProject;
    markerToProject.push_back(
                IndexPairing (Instant::ALL, {idxSegmentBow, stringBridgeIdx, PLANE::XZ}));

    // Stay on one string and have a good direction of the bow
    std::vector<IndexPairing> alignWithMarkersReferenceFrame;
    alignWithMarkersReferenceFrame.push_back(IndexPairing(Instant::ALL,
            {idxSegmentBow, AXIS::X, stringNeckIdx, stringBridgeIdx,
             AXIS::Y, idxLowStringBound, idxHighStringBound, AXIS::Y}));

    // No need to aligning with markers
    std::vector<IndexPairing> alignWithMarkers;

    // No need to aligning two segments
    std::vector<IndexPairing> axesToAlign;




    // From here, unless one wants to fundamentally change the problem,
    // they should not change anything
    casadi::MX V;
    BoundaryConditions vBounds;
    InitialConditions vInit;
    std::vector<casadi::MX> g;
    BoundaryConditions gBounds;
    casadi::MX J;
    prepareMusculoSkeletalNLP(probSize, odeSolver, uBounds, uInit, xBounds, xInit,
                              markersToPair, markerToProject, axesToAlign, alignWithMarkers, alignWithMarkersReferenceFrame,
                              useCyclicObjective, useCyclicConstraint, objectiveFunctions,
                              V, vBounds, vInit, g, gBounds, J);

    // Online visualization
    AnimationCallback animCallback(visu, V, g, probSize, 10);

    // Optimize
    clock_t start = clock();
    std::vector<double> V_opt = solveProblemWithIpopt(V, vBounds, vInit, J, g, gBounds, probSize, animCallback);
    clock_t end=clock();

    // Get the optimal state trajectory
    finalizeSolution(V_opt, probSize, optimizationName);

    // ---------- FINALIZE  ------------ //
    double time_exec(double(end - start)/CLOCKS_PER_SEC);
//    std::cout << "Execution time = " << time_exec<<std::endl;

    while(animCallback.isActive()){}
    return 0;
}
