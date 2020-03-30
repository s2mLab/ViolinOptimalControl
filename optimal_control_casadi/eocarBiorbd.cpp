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
    probSize.ns = 50;
    probSize.dt = probSize.tf/probSize.ns; // length of a control interval

    // Chose the ODE solver
    ODE_SOLVER odeSolver(ODE_SOLVER::RK);

    // Chose the objective functions
    std::vector<std::pair<void (*)(const ProblemSize&,
                         const std::vector<casadi::MX>&,
                         const std::vector<casadi::MX>&,
                         double,
                         casadi::MX&), double>> objectiveFunctions;
    objectiveFunctions.push_back(std::make_pair(minimizeTorqueControls, 10));
    objectiveFunctions.push_back(std::make_pair(minimizeMuscleControls, 10));
    objectiveFunctions.push_back(std::make_pair(minimizeStates, 1./100));

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

        if (i == 0){
            xInit.val.push_back(1.5);
        } else {
            xInit.val.push_back(0);
        }
    };
    double velLim(15);
    for (unsigned int i=0; i<m.nbQdot(); ++i) {
        xBounds.starting_min.push_back(0);
        xBounds.min.push_back(-velLim);
        xBounds.end_min.push_back(0);

        xBounds.starting_max.push_back(0);
        xBounds.max.push_back(velLim);
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

    // If the movement is cyclic
    bool useCyclicObjective = false;
    bool useCyclicConstraint = false;

    // Start at the starting point and finish at the ending point
    std::vector<IndexPairing> markersToPair;
    markersToPair.push_back(IndexPairing(Instant::START, {0, 1}));
    markersToPair.push_back(IndexPairing(Instant::MID, {0, 2}));
    markersToPair.push_back(IndexPairing(Instant::END, {0, 1}));

    // Always point towards the point(3)
    std::vector<IndexPairing> markerToProject;
//    markerToProject.push_back(IndexPairing (Instant::ALL, {0, 3, PLANE::XZ}));

    // Always point upward
    std::vector<IndexPairing> axesToAlign;
//    axesToAlign.push_back(IndexPairing(Instant::INTERMEDIATES, {0, AXIS::X, 1, AXIS::MINUS_Y}));

    // Always point in line with a given vector described by markers
    std::vector<IndexPairing> alignWithMarkers;
//    alignWithMarkers.push_back(IndexPairing(Instant::INTERMEDIATES, {0, AXIS::X, 2, 3}));

    // Always have the segment aligned with a certain system of axes
    std::vector<IndexPairing> alignWithMarkersReferenceFrame;
//    alignWithMarkersReferenceFrame.push_back(IndexPairing(Instant::ALL,
//            {0, AXIS::X, 1, 2, AXIS::Y, 1, 3, AXIS::X}));

    // Always point toward a specific IMU
    std::vector<IndexPairing> alignWithCustomRT;
    alignWithCustomRT.push_back(IndexPairing(Instant::ALL,{0, 0}));



    // From here, unless one wants to fundamentally change the problem,
    // they should not change anything
    casadi::MX V;
    BoundaryConditions vBounds;
    InitialConditions vInit;
    std::vector<casadi::MX> g;
    BoundaryConditions gBounds;
    casadi::MX J;
    casadi::Function dynamics;
    prepareMusculoSkeletalNLP(probSize, odeSolver, uBounds, uInit, xBounds, xInit,
                              markersToPair, markerToProject, axesToAlign,
                              alignWithMarkers, alignWithMarkersReferenceFrame, alignWithCustomRT,
                              useCyclicObjective, useCyclicConstraint, objectiveFunctions,
                              V, vBounds, vInit, g, gBounds, J, dynamics);

    // Optimize
    AnimationCallback animCallback(visu, V, g, probSize, 10, dynamics);
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
