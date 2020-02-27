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
                         int,
                         casadi::MX&), int>> objectiveFunctions;
    objectiveFunctions.push_back(std::make_pair(minimizeTorqueControls, 1000));
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
        xBounds.starting_min.push_back(-100);
        xBounds.min.push_back(-100);
        xBounds.end_min.push_back(-100);

        xBounds.starting_max.push_back(100);
        xBounds.max.push_back(100);
        xBounds.end_max.push_back(100);

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
    bool useCyclicObjective = true;
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
    std::cout<<"Execution time: "<<time_exec<<std::endl;

    while(animCallback.isActive()){}
    return 0;
}
