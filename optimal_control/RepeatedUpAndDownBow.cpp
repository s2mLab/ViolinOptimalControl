#include <iostream>
#include <fstream>
#include <memory>
#include <time.h>
#include <acado_optimal_control.hpp>
#include <bindings/acado_gnuplot/gnuplot_window.hpp>

#include "biorbd.h"
#include "includes/utils.h"
#include "includes/dynamics.h"
#include "includes/constraints.h"
#include "includes/objectives.h"

#include "includes/violinStringConfig.h"

biorbd::Model m("../../models/BrasViolon.bioMod");
#include "includes/biorbd_initializer.h"

static int idxSegmentBow(8);
static int tagBowFrog(16);
static int tagBowTip(18);
static int tagViolinBString(38);
static int tagViolinEString(34);
static int tagViolinAString(35);
static int tagViolinDString(36);
static int tagViolinGString(37);
static int tagViolinCString(39);

const double t_Start = 0.0;
const double t_End = 0.5;
const int nPoints(31);

const ViolinStringNames stringPlayed(ViolinStringNames::E);
const bool useFileToInit(false);
const int nBowing(1);
const int nBowingInInitialization(1);

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

const std::string resultsPath("../Results/");
const std::string initializePath("../Initialisation/");
const std::string optimizationName("RepeatedUpAndDownBow");

const std::string diffStateResultsFileName(resultsPath + "States" + optimizationName + ".txt");
const std::string controlResultsFileName(resultsPath + "Controls" + optimizationName + ".txt");
const std::string diffStateWithoutBrackets(resultsPath + "StatesNo[]" + optimizationName + ".txt");
const std::string controlWithoutBrackets(resultsPath + "ControlNo[]" + optimizationName + ".txt");



USING_NAMESPACE_ACADO
int  main ()
{
    clock_t start = clock();
    std::cout << "nb de muscles: " << nMus << std::endl;
    std::cout << "nb de torques: " << nTau << std::endl;
    std::cout << "nb de marqueurs: " << nMarkers << std::endl;
    initializeMuscleStates();

    // ----------- DEFINE OCP ------------- //
    OCP ocp(t_Start, t_End, nPoints);
    CFunction residualTorque(1, residualTorquesSquare);
    CFunction muscleActivation(1, muscleActivationsSquare);
    CFunction bowOnString(3, stringToPlayObjective);
    CFunction F( nQ+nQdot, forwardDynamics_noContact);
    DifferentialEquation f ;

    // --------- DEFINE SOME CONSTRAINT FUNCTIONS --------- //
    CFunction markerBowFrog(3, markerPosition);
    markerBowFrog.setUserData(static_cast<void*>(&tagBowFrog));
    CFunction markerBowTip(3, markerPosition);
    markerBowTip.setUserData(static_cast<void*>(&tagBowTip));
    CFunction markerViolinString(3, markerPosition);
    int bowAndViolinMarkersToAlign[4];
    CFunction bowDirection(2, projectOnXzPlane);
    int idxProjectViolinBridgeInBow[2];

    bowAndViolinMarkersToAlign[0] = tagBowFrog;
    bowAndViolinMarkersToAlign[1] = tagBowTip;
    if (stringPlayed == ViolinStringNames::E) {
        markerViolinString.setUserData(static_cast<void*>(&tagViolinEString));
        idxProjectViolinBridgeInBow[0] = tagViolinEString;
        bowAndViolinMarkersToAlign[2] = tagViolinBString;
        bowAndViolinMarkersToAlign[3] = tagViolinAString;
    }
    else if (stringPlayed == ViolinStringNames::A) {
        markerViolinString.setUserData(static_cast<void*>(&tagViolinAString));
        idxProjectViolinBridgeInBow[0] = tagViolinAString;
        bowAndViolinMarkersToAlign[2] = tagViolinEString;
        bowAndViolinMarkersToAlign[3] = tagViolinDString;
    }
    else if (stringPlayed == ViolinStringNames::D) {
        markerViolinString.setUserData(static_cast<void*>(&tagViolinDString));
        idxProjectViolinBridgeInBow[0] = tagViolinDString;
        bowAndViolinMarkersToAlign[2] = tagViolinAString;
        bowAndViolinMarkersToAlign[3] = tagViolinGString;
    }
    else if (stringPlayed == ViolinStringNames::D) {
        markerViolinString.setUserData(static_cast<void*>(&tagViolinGString));
        idxProjectViolinBridgeInBow[0] = tagViolinGString;
        bowAndViolinMarkersToAlign[2] = tagViolinDString;
        bowAndViolinMarkersToAlign[3] = tagViolinCString;
    }
    else {
        throw std::runtime_error("Wrong choice of stringPlayed");
    }
    idxProjectViolinBridgeInBow[1] = idxSegmentBow;

    bowDirection.setUserData(static_cast<void*>(idxProjectViolinBridgeInBow));
    bowOnString.setUserData(static_cast<void*>(bowAndViolinMarkersToAlign));

    // Get the ranges (limits of DoF)
    std::vector<biorbd::utils::Range> ranges;
    for (unsigned int i=0; i<m.nbSegment(); ++i){
        std::vector<biorbd::utils::Range> segRanges(m.segment(i).ranges());
        for(unsigned int j=0; j<segRanges.size(); ++j){
            ranges.push_back(segRanges[j]);
        }
    }

    // ---------- INITIALIZATION ---------- //
    std::vector<DifferentialState> x;
    std::vector<Control> control;
    std::vector<IntermediateState> is;

    // ---------- PHASES ---------- //
    // Each phase is a up/down bow (hence the nBowing*2)

    // Dynamic and path constraints
    for (unsigned int p=0; p<nBowing*2; ++p){
        x.push_back(DifferentialState("",nQ+nQdot,1));
        control.push_back(Control("", nMus + nTau, 1));
        is.push_back(IntermediateState("", nQ+nQdot+nMus+nTau, 1));

        for (unsigned int i = 0; i < nQ; ++i)
            is[p](i) = x[p](i);
        for (unsigned int i = 0; i < nQdot; ++i)
            is[p](i+nQ) = x[p](i+nQ);
        for (unsigned int i = 0; i < nMus; ++i)
            is[p](i+nQ+nQdot) = control[p](i);
        for (unsigned int i = 0; i < nTau; ++i)
            is[p](i+nQ+nQdot+nMus) = control[p](i+nMus);

        // Dynamics
        (f << dot(x[p])) == F(is[p]);

        // ------------ PATH CONSTRAINTS ----------- //
        // Bound for the controls
        for (unsigned int i=0; i<nMus; ++i){
            ocp.subjectTo(0.01 <= control[p](i) <= 1);
        }
        for (unsigned int i=0; i<nTau; ++i){
            ocp.subjectTo(-100 <= control[p](i+nMus) <= 100);
        }

        // Bound for the states
        for (unsigned int i=0; i<nQ; ++i){
            ocp.subjectTo(ranges[i].min() <= x[p](0) <= ranges[i].max());
        }
        for (unsigned int j=0; j<nQdot; ++j) {
            ocp.subjectTo(-50 <= x[p](nQ + j) <= 50);
        }

        // Movement starts at frog, goes to tip and comes back to frog
        if (p == 0){
            ocp.subjectTo(AT_START, markerBowFrog(x[p]) - markerViolinString(x[p]) == 0.0 );
            ocp.subjectTo(AT_END, markerBowTip(x[p]) - markerViolinString(x[p]) == 0.0 );
        }
        else {
            ocp.subjectTo( 0.0, x[p], -x[p-1], 0.0 );
            ocp.subjectTo( 0.0, x[p-1], -x[p], 0.0 );
        }

        // The bow must remain perpendicular to the violin on the chosen string
        for (int i = 1; i < nPoints - 1; ++i) {
            ocp.subjectTo(i, bowDirection(x[p]) == 0.0);
            ocp.subjectTo(i, bowOnString(x[p]) == 0.0);
        }
    }
    ocp.subjectTo(f);

    // ------------ OBJECTIVE ----------- //
    Expression sumLagrange = residualTorque(control[0]) + muscleActivation(control[0]);// + bowOnString(x[0]) * 100;
    for(unsigned int p=1; p<nBowing*2; ++p)
        sumLagrange += residualTorque(control[p]) + muscleActivation(control[p]); // + bowOnString(x[p]) * 100;
    ocp.minimizeLagrangeTerm( sumLagrange );


    // ---------- OPTIMIZATION  ------------ //
    OptimizationAlgorithm  algorithm(ocp);
    algorithm.set(MAX_NUM_ITERATIONS, 1000);
    algorithm.set(INTEGRATOR_TYPE, INT_RK45);
    algorithm.set(HESSIAN_APPROXIMATION, FULL_BFGS_UPDATE);
    algorithm.set(KKT_TOLERANCE, 1e-4);


    // ---------- INITIAL SOLUTION ---------- //
    VariablesGrid *u_init;
    VariablesGrid *x_init;

    if (useFileToInit) {
        u_init = new VariablesGrid(nBowingInInitialization*2*(nTau + nMus), Grid(t_Start, t_End, nPoints+1));
        x_init = new VariablesGrid(nBowingInInitialization*2*(nQ+nQdot), Grid(t_Start, t_End, nPoints+1));

        removeSquareBracketsInFile(controlResultsFileName, controlWithoutBrackets);
        removeSquareBracketsInFile(diffStateResultsFileName, diffStateWithoutBrackets);

        *u_init = readControls(controlWithoutBrackets, nPoints, nBowingInInitialization*2, t_Start, t_End);
        *x_init = readStates(diffStateWithoutBrackets, nPoints, nBowingInInitialization*2, t_Start, t_End);
    }
    else {
        u_init = new VariablesGrid(nBowingInInitialization*2*(nTau + nMus), Grid(t_Start, t_End, 2));
        x_init = new VariablesGrid(nBowingInInitialization*2*(nQ+nQdot), Grid(t_Start, t_End, 2));

        // Initialize controls
        for(unsigned int i=0; i<nBowingInInitialization*2; ++i){
            for(unsigned int j=0; j<nMus; ++j){
                (*u_init)(0, i*(nMus+nTau) + j ) = 0.2;
                (*u_init)(1, i*(nMus+nTau) + j ) = 0.2;
            }
            for(unsigned int j=0; j<nTau; ++j){
                (*u_init)(0, i*(nMus+nTau) + nMus + j ) = 0.01;
                (*u_init)(1, i*(nMus+nTau) + nMus + j ) = 0.01;
            }
        }

        // Initialize states
        std::vector<double> initQFrogOnSelectedString;
        std::vector<double> initQTipOnSelectedString;
        if (stringPlayed == ViolinStringNames::E) {
            initQFrogOnSelectedString = initQFrogOnEString;
            initQTipOnSelectedString = initQTipOnEString;
        }
        else if (stringPlayed == ViolinStringNames::A){
            initQFrogOnSelectedString = initQFrogOnAString;
            initQTipOnSelectedString = initQTipOnAString;
        }
        else if (stringPlayed == ViolinStringNames::D){
            initQFrogOnSelectedString = initQFrogOnDString;
            initQTipOnSelectedString = initQTipOnDString;
        }
        else if (stringPlayed == ViolinStringNames::G){
            initQFrogOnSelectedString = initQFrogOnGString;
            initQTipOnSelectedString = initQTipOnGString;
        }

        // Q
        for(unsigned int i=0; i < nBowingInInitialization; ++i){
            // BowFrog and tip on ViolinBridge
            for (unsigned int j=0; j<nQ; ++j){
                (*x_init)(0, 2*i*(nQ+nQdot)+j) = initQFrogOnSelectedString[j];
                (*x_init)(1, 2*i*(nQ+nQdot)+j) = initQTipOnSelectedString[j];
                (*x_init)(0, ((2*i)+1)*(nQ+nQdot)+j) = initQTipOnSelectedString[j];
                (*x_init)(1, ((2*i)+1)*(nQ+nQdot)+j) = initQFrogOnSelectedString[j];
            }
        }

        // Qdot
        for(unsigned int i=0; i<nBowingInInitialization*2; ++i){
            for(unsigned int j=0; j<nQdot; ++j){
                (*x_init)(0, i*(nQ+nQdot) + nQ + j) = 0.01;
                (*x_init)(1, i*(nQ+nQdot) + nQ + j) = 0.01;
            }
        }
    }

    if (nBowing > nBowingInInitialization) {
        // Duplicate the initial solution if needed
        VariablesGrid x_init_expanded, u_init_expanded, copy_init_1;


        duplicateElements(nBowing*2, nBowingInInitialization*2, (nQ + nQdot) * 2, 0,
                          *x_init, copy_init_1, x_init_expanded);
        duplicateElements(nBowing*2, nBowingInInitialization*2, (nMus + nTau) * 2, 1,
                          *u_init, copy_init_1, u_init_expanded);

        algorithm.initializeControls(u_init_expanded);
        algorithm.initializeDifferentialStates(x_init_expanded);
    }
    else {
        algorithm.initializeControls(*u_init);
        algorithm.initializeDifferentialStates(*x_init);
    }


    // ---------- SOLVING THE PROBLEM ---------- //
    algorithm.solve();


    // ---------- STORING THE RESULTS ---------- //
    createTreePath(resultsPath);
    algorithm.getDifferentialStates(diffStateResultsFileName.c_str());
    algorithm.getControls(controlResultsFileName.c_str());


    // ---------- PLOTING TIME ---------- //
    clock_t end=clock();
    double time_exec(double(end - start)/CLOCKS_PER_SEC);
    std::cout << "Execution time: " << time_exec << std::endl;


    // ---------- EXIT ---------- //
    return 0;
}
