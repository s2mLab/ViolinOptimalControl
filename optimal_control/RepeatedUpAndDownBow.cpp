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

 #define USE_INIT_FILE
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

const ViolinStringNames stringName(ViolinStringNames::E);
const bool useFileToInit(false);
const int nBowing(1);
const int nBowingInInitialization(1);

// The init on string were determined using the "find_initial_pose.py" script
const std::vector<double> initQFrogOnGString =
{-0.00948486, 0.06259299, 0.99964932, 0.92035463, 1.40957673, 0.32581681, -0.07523013, -0.76109885};
const std::vector<double> initQTipOnGString =
{0.01113298, -0.61721062, 0.96989367, 0.59865875, 0.19520906, 0.11549791, 0.10830705, 0.54975026};
const std::vector<double> initQFrogOnDString =
{0.0239408, 0.08831102, 0.95293047, 0.94194847, 1.4724044, 0.30109847, -0.44859286, -0.46923365};
const std::vector<double> initQTipOnDString =
{0.08263522, -0.62539549, 0.90233634, 0.62698962, 0.22359432, 0.12548774, 0.09448064, 0.54877327};
const std::vector<double> initQFrogOnAString =
{0.01018357, 0.09299291, 0.88991844, 0.931988, 1.4649005, 0.25007886, -0.34189658, -0.54073149};
const std::vector<double> initQTipOnAString =
{0.09638825, -0.53412493, 0.81893825, 0.69326372, 0.22566753, 0.11652397, 0.18524205, 0.46584988};
const std::vector<double> initQFrogOnEString =
{ 0.08400899, 0.09984273, 0.79351699, 0.90026544, 1.45634165, 0.32713986, -0.25263593, -0.64335862};
const std::vector<double> initQTipOnEString =
{0.07910913, -0.45011153, 0.778877, 0.73878697, 0.21872682, 0.10636272, 0.16720347, 0.48748324};

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
    CFunction bowDirection(3, bowDirectionAgainstViolin);
    CFunction F( nQ+nQdot, forwardDynamics_noContact);
    DifferentialEquation f ;

    // --------- DEFINE SOME PATH CONSTRAINTS --------- //
    CFunction markerBowFrog(3, markerPosition);
    markerBowFrog.setUserData(static_cast<void*>(&tagBowFrog));
    CFunction markerBowTip(3, markerPosition);
    markerBowTip.setUserData(static_cast<void*>(&tagBowTip));
    CFunction markerViolinString(3, markerPosition);
    int bowAndViolinMarkersToAlign[4];
    CFunction violinBridgeInBowRT(2, projectOnXzPlane);
    int idxProjectViolinBridgeInBow[2];

    bowAndViolinMarkersToAlign[0] = tagBowFrog;
    bowAndViolinMarkersToAlign[1] = tagBowTip;
    switch (stringName) {
    case ViolinStringNames::E:
        markerViolinString.setUserData(static_cast<void*>(&tagViolinEString));
        idxProjectViolinBridgeInBow[0] = tagViolinEString;
        bowAndViolinMarkersToAlign[2] = tagViolinBString;
        bowAndViolinMarkersToAlign[3] = tagViolinAString;
        break;
    case ViolinStringNames::A:
        markerViolinString.setUserData(static_cast<void*>(&tagViolinAString));
        idxProjectViolinBridgeInBow[0] = tagViolinAString;
        bowAndViolinMarkersToAlign[2] = tagViolinEString;
        bowAndViolinMarkersToAlign[3] = tagViolinDString;
        break;
    case ViolinStringNames::D:
        markerViolinString.setUserData(static_cast<void*>(&tagViolinDString));
        idxProjectViolinBridgeInBow[0] = tagViolinDString;
        bowAndViolinMarkersToAlign[2] = tagViolinAString;
        bowAndViolinMarkersToAlign[3] = tagViolinGString;
        break;
    case ViolinStringNames::G:
        markerViolinString.setUserData(static_cast<void*>(&tagViolinGString));
        idxProjectViolinBridgeInBow[0] = tagViolinGString;
        bowAndViolinMarkersToAlign[2] = tagViolinDString;
        bowAndViolinMarkersToAlign[3] = tagViolinCString;
        break;
    }
    idxProjectViolinBridgeInBow[1] = idxSegmentBow;

    violinBridgeInBowRT.setUserData(static_cast<void*>(idxProjectViolinBridgeInBow));
    bowDirection.setUserData(static_cast<void*>(bowAndViolinMarkersToAlign));

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
    // Each phase is a up/down bow (hence the *2)
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

        // ------------ CONSTRAINTS ----------- //
        // Dynamics
        (f << dot(x[p])) == F(is[p]);

        // Controls constraints
        for (unsigned int i=0; i<nMus; ++i){
            ocp.subjectTo(0.01 <= control[p](i) <= 1);
        }
        for (unsigned int i=0; i<nTau; ++i){
            ocp.subjectTo(-100 <= control[p](i+nMus) <= 100);
        }

        // Path constraints
        if(p==0) {                            
                ocp.subjectTo(
                            AT_START, markerBowFrog(x[p]) - markerViolinString(x[p])
                            == 0.0 );
                ocp.subjectTo(
                            AT_END, markerBowTip(x[p]) - markerViolinString(x[p])
                            == 0.0 );
                }
        else {
            ocp.subjectTo( 0.0, x[p], -x[p-1], 0.0 );
            ocp.subjectTo( 0.0, x[p-1], -x[p], 0.0 );
        }
        for (int i = 1; i < nPoints-1; ++i) {
            ocp.subjectTo(i, violinBridgeInBowRT(x[p]) == 0.0);
    //        ocp.subjectTo(i, bowDirection(x[p]) == 0.0);
        }

        // Set the limit of the degrees of freedom
        for (unsigned int i=0; i<ranges.size(); ++i){
            ocp.subjectTo(ranges[i].min() <= x[p](0) <= ranges[i].max());
        }

        for (unsigned int j=0; j<nQdot; ++j) {
            ocp.subjectTo(-50 <= x[p](nQ + j) <= 50);
        }

    }
    ocp.subjectTo(f);

    // ------------ OBJECTIVE ----------- //
    Expression sumLagrange = residualTorque(control[0])+ muscleActivation(control[0]) + bowDirection(x[0]);
    for(unsigned int p=1; p<nBowing*2; ++p)
        sumLagrange += residualTorque(control[p]) + muscleActivation(control[p]) + bowDirection(x[p]);
    ocp.minimizeLagrangeTerm( sumLagrange );

    // ---------- OPTIMIZATION  ------------ //
    OptimizationAlgorithm  algorithm(ocp) ;
    algorithm.set(MAX_NUM_ITERATIONS, 1000);
    algorithm.set(INTEGRATOR_TYPE, INT_RK45);
    algorithm.set(HESSIAN_APPROXIMATION, CONSTANT_HESSIAN);
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
        if (stringName == ViolinStringNames::E) {
            initQFrogOnSelectedString = initQFrogOnEString;
            initQTipOnSelectedString = initQTipOnEString;
        }
        else if (stringName == ViolinStringNames::A){
            initQFrogOnSelectedString = initQFrogOnAString;
            initQTipOnSelectedString = initQTipOnAString;
        }
        else if (stringName == ViolinStringNames::D){
            initQFrogOnSelectedString = initQFrogOnDString;
            initQTipOnSelectedString = initQTipOnDString;
        }
        else if (stringName == ViolinStringNames::G){
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

    // ---------- PLOTING ---------- //
    clock_t end=clock();
    double time_exec(double(end - start)/CLOCKS_PER_SEC);
    std::cout << "Execution time: " << time_exec << std::endl;

    // ---------- EXIT ---------- //
    return 0;
}
