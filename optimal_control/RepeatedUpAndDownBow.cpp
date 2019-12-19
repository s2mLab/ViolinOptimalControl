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

static int idxSegmentBow = 8;
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

        ocp.subjectTo(-M_PI/8 <= x[p](0) <= 0.1);
        ocp.subjectTo(-M_PI/2 <= x[p](1) <= 0.1);
        ocp.subjectTo(-M_PI/4 <= x[p](2) <= M_PI);
        ocp.subjectTo(-M_PI/2 <= x[p](3) <= M_PI/2);
        ocp.subjectTo(-0.1  <= x[p](4) <= M_PI);
        ocp.subjectTo(-M_PI/4 <= x[p](5) <= M_PI/4);
        ocp.subjectTo(-M_PI   <= x[p](6) <= M_PI);
        ocp.subjectTo(-M_PI/4 <= x[p](7) <= M_PI/4);

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
        switch (stringName) {
        case ViolinStringNames::E:
            for(unsigned int i=0; i < nBowingInInitialization; ++i){
                // BowFrog on ViolinBridge
                (*x_init)(0, 2*i*(nQ+nQdot)+0) = -0.2725;
                (*x_init)(0, 2*i*(nQ+nQdot)+1) = -0.4238;
                (*x_init)(0, 2*i*(nQ+nQdot)+2) = 1.2047;
                (*x_init)(0, 2*i*(nQ+nQdot)+3) = 0.7291;
                (*x_init)(0, 2*i*(nQ+nQdot)+4) = 1.4200;
                (*x_init)(0, 2*i*(nQ+nQdot)+5) = -0.1819;
                (*x_init)(0, 2*i*(nQ+nQdot)+6) = -1.7549;
                (*x_init)(0, 2*i*(nQ+nQdot)+7) = -0.5246;

                // BowTip on ViolinBridge
                (*x_init)(1, 2*i*(nQ+nQdot)+0) = -0.0075;
                (*x_init)(1, 2*i*(nQ+nQdot)+1) = -0.3963;
                (*x_init)(1, 2*i*(nQ+nQdot)+2) = 0.8140;
                (*x_init)(1, 2*i*(nQ+nQdot)+3) = 1.1841;
                (*x_init)(1, 2*i*(nQ+nQdot)+4) = 0.2775;
                (*x_init)(1, 2*i*(nQ+nQdot)+5) = -0.2605;
                (*x_init)(1, 2*i*(nQ+nQdot)+6) = -1.7161;
                (*x_init)(1, 2*i*(nQ+nQdot)+7) = 0.6331;

                // BowTip on ViolinBridge
                (*x_init)(0, ((2*i)+1)*(nQ+nQdot)+0) = -0.0075;
                (*x_init)(0, ((2*i)+1)*(nQ+nQdot)+1) = -0.3963;
                (*x_init)(0, ((2*i)+1)*(nQ+nQdot)+2) = 0.8140;
                (*x_init)(0, ((2*i)+1)*(nQ+nQdot)+3) = 1.1841;
                (*x_init)(0, ((2*i)+1)*(nQ+nQdot)+4) = 0.2775;
                (*x_init)(0, ((2*i)+1)*(nQ+nQdot)+5) = -0.2605;
                (*x_init)(0, ((2*i)+1)*(nQ+nQdot)+6) = -1.7161;
                (*x_init)(0, ((2*i)+1)*(nQ+nQdot)+7) = 0.6331;

                // BowFrog on ViolinBridge
                (*x_init)(1, ((2*i)+1)*(nQ+nQdot)+0) = -0.2725;
                (*x_init)(1, ((2*i)+1)*(nQ+nQdot)+1) = -0.4238;
                (*x_init)(1, ((2*i)+1)*(nQ+nQdot)+2) = 1.2047;
                (*x_init)(1, ((2*i)+1)*(nQ+nQdot)+3) = 0.7291;
                (*x_init)(1, ((2*i)+1)*(nQ+nQdot)+4) = 1.4200;
                (*x_init)(1, ((2*i)+1)*(nQ+nQdot)+5) = -0.1819;
                (*x_init)(1, ((2*i)+1)*(nQ+nQdot)+6) = -1.7549;
                (*x_init)(1, ((2*i)+1)*(nQ+nQdot)+7) = -0.5246;
            }
            break;

        case ViolinStringNames::A:
            for(unsigned int i=0; i < nBowingInInitialization; ++i){

                // BowFrog on ViolinBridge
                (*x_init)(0, 2*i*(nQ+nQdot)+0) = -0.1374;
                (*x_init)(0, 2*i*(nQ+nQdot)+1) = -0.0546;
                (*x_init)(0, 2*i*(nQ+nQdot)+2) = 1.0780;
                (*x_init)(0, 2*i*(nQ+nQdot)+3) = 1.0717;
                (*x_init)(0, 2*i*(nQ+nQdot)+4) = 1.3993;
                (*x_init)(0, 2*i*(nQ+nQdot)+5) = -0.7248;
                (*x_init)(0, 2*i*(nQ+nQdot)+6) = -0.7638;
                (*x_init)(0, 2*i*(nQ+nQdot)+7) = -0.5774;

                // BowTip on ViolinBridge
                (*x_init)(1, 2*i*(nQ+nQdot)+0) = 0.1000;
                (*x_init)(1, 2*i*(nQ+nQdot)+1) = -0.6409;
                (*x_init)(1, 2*i*(nQ+nQdot)+2) = 0.9841;
                (*x_init)(1, 2*i*(nQ+nQdot)+3) = 0.8485;
                (*x_init)(1, 2*i*(nQ+nQdot)+4) = 0.1246;
                (*x_init)(1, 2*i*(nQ+nQdot)+5) = 0.7308;
                (*x_init)(1, 2*i*(nQ+nQdot)+6) = -0.8842;
                (*x_init)(1, 2*i*(nQ+nQdot)+7) = 0.7854;

                // BowTip on ViolinBridge
                (*x_init)(0, ((2*i)+1)*(nQ+nQdot)+0) = 0.1000;
                (*x_init)(0, ((2*i)+1)*(nQ+nQdot)+1) = -0.6409;
                (*x_init)(0, ((2*i)+1)*(nQ+nQdot)+2) = 0.9841;
                (*x_init)(0, ((2*i)+1)*(nQ+nQdot)+3) = 0.8485;
                (*x_init)(0, ((2*i)+1)*(nQ+nQdot)+4) = 0.1246;
                (*x_init)(0, ((2*i)+1)*(nQ+nQdot)+5) = 0.7308;
                (*x_init)(0, ((2*i)+1)*(nQ+nQdot)+6) = -0.8842;
                (*x_init)(0, ((2*i)+1)*(nQ+nQdot)+7) = 0.7854;

                // BowFrog on ViolinBridge
                (*x_init)(1, ((2*i)+1)*(nQ+nQdot)+0) = -0.1374;
                (*x_init)(1, ((2*i)+1)*(nQ+nQdot)+1) = -0.0546;
                (*x_init)(1, ((2*i)+1)*(nQ+nQdot)+2) = 1.0780;
                (*x_init)(1, ((2*i)+1)*(nQ+nQdot)+3) = 1.0717;
                (*x_init)(1, ((2*i)+1)*(nQ+nQdot)+4) = 1.3993;
                (*x_init)(1, ((2*i)+1)*(nQ+nQdot)+5) = -0.7248;
                (*x_init)(1, ((2*i)+1)*(nQ+nQdot)+6) = -0.7638;
                (*x_init)(1, ((2*i)+1)*(nQ+nQdot)+7) = -0.5774;
            }
            break;

        case ViolinStringNames::D:
            for(unsigned int i=0; i < nBowingInInitialization; ++i){

                // BowFrog on ViolinBridge
                (*x_init)(0, 2*i*(nQ+nQdot)+0) = -0.2725;
                (*x_init)(0, 2*i*(nQ+nQdot)+1) = -0.0221;
                (*x_init)(0, 2*i*(nQ+nQdot)+2) = 1.2267;
                (*x_init)(0, 2*i*(nQ+nQdot)+3) = 0.7916;
                (*x_init)(0, 2*i*(nQ+nQdot)+4) = 1.3559;
                (*x_init)(0, 2*i*(nQ+nQdot)+5) = 0.7579;
                (*x_init)(0, 2*i*(nQ+nQdot)+6) = -1.7638;
                (*x_init)(0, 2*i*(nQ+nQdot)+7) = -0.3637;

                // BowTip on ViolinBridge
                (*x_init)(1, 2*i*(nQ+nQdot)+0) = -0.1828;
                (*x_init)(1, 2*i*(nQ+nQdot)+1) = 0.1000;
                (*x_init)(1, 2*i*(nQ+nQdot)+2) = 0.9532;
                (*x_init)(1, 2*i*(nQ+nQdot)+3) = 0.4593;
                (*x_init)(1, 2*i*(nQ+nQdot)+4) = 0.0422;
                (*x_init)(1, 2*i*(nQ+nQdot)+5) = -0.2124;
                (*x_init)(1, 2*i*(nQ+nQdot)+6) = -0.5838;
                (*x_init)(1, 2*i*(nQ+nQdot)+7) = 0.6979;

                // BowTip on ViolinBridge
                (*x_init)(0, ((2*i)+1)*(nQ+nQdot)+0) = -0.1828;
                (*x_init)(0, ((2*i)+1)*(nQ+nQdot)+1) = 0.1000;
                (*x_init)(0, ((2*i)+1)*(nQ+nQdot)+2) = 0.9532;
                (*x_init)(0, ((2*i)+1)*(nQ+nQdot)+3) = 0.4593;
                (*x_init)(0, ((2*i)+1)*(nQ+nQdot)+4) = 0.0422;
                (*x_init)(0, ((2*i)+1)*(nQ+nQdot)+5) = -0.2124;
                (*x_init)(0, ((2*i)+1)*(nQ+nQdot)+6) = -0.5838;
                (*x_init)(0, ((2*i)+1)*(nQ+nQdot)+7) = 0.6979;

                // BowFrog on ViolinBridge
                (*x_init)(1, ((2*i)+1)*(nQ+nQdot)+0) = -0.2725;
                (*x_init)(1, ((2*i)+1)*(nQ+nQdot)+1) = -0.0221;
                (*x_init)(1, ((2*i)+1)*(nQ+nQdot)+2) = 1.2267;
                (*x_init)(1, ((2*i)+1)*(nQ+nQdot)+3) = 0.7916;
                (*x_init)(1, ((2*i)+1)*(nQ+nQdot)+4) = 1.3559;
                (*x_init)(1, ((2*i)+1)*(nQ+nQdot)+5) = 0.7579;
                (*x_init)(1, ((2*i)+1)*(nQ+nQdot)+6) = -1.7638;
                (*x_init)(1, ((2*i)+1)*(nQ+nQdot)+7) = -0.3637;
            }
            break;

        case ViolinStringNames::G:
            for(unsigned int i=0; i < nBowingInInitialization; ++i){

                // BowFrog on ViolinBridge               
                (*x_init)(0, 2*i*(nQ+nQdot)+0) = 0.0611;
                (*x_init)(0, 2*i*(nQ+nQdot)+1) = -0.5424;
                (*x_init)(0, 2*i*(nQ+nQdot)+2) = 1.2640;
                (*x_init)(0, 2*i*(nQ+nQdot)+3) = 0.5149;
                (*x_init)(0, 2*i*(nQ+nQdot)+4) = 1.3547;
                (*x_init)(0, 2*i*(nQ+nQdot)+5) = 0.7531;
                (*x_init)(0, 2*i*(nQ+nQdot)+6) = -1.7864;
                (*x_init)(0, 2*i*(nQ+nQdot)+7) = -0.4478;

                // BowTip on ViolinBridge
                (*x_init)(1, 2*i*(nQ+nQdot)+0) = 0.0096;
                (*x_init)(1, 2*i*(nQ+nQdot)+1) = -0.5954;
                (*x_init)(1, 2*i*(nQ+nQdot)+2) = 1.0852;
                (*x_init)(1, 2*i*(nQ+nQdot)+3) = 0.4258;
                (*x_init)(1, 2*i*(nQ+nQdot)+4) = 0.0566;
                (*x_init)(1, 2*i*(nQ+nQdot)+5) = 0.5255;
                (*x_init)(1, 2*i*(nQ+nQdot)+6) = -0.7440;
                (*x_init)(1, 2*i*(nQ+nQdot)+7) = 0.6530;

                // BowTip on ViolinBridge
                (*x_init)(0, ((2*i)+1)*(nQ+nQdot)+0) = 0.0096;
                (*x_init)(0, ((2*i)+1)*(nQ+nQdot)+1) = -0.5954;
                (*x_init)(0, ((2*i)+1)*(nQ+nQdot)+2) = 1.0852;
                (*x_init)(0, ((2*i)+1)*(nQ+nQdot)+3) = 0.4258;
                (*x_init)(0, ((2*i)+1)*(nQ+nQdot)+4) = 0.0566;
                (*x_init)(0, ((2*i)+1)*(nQ+nQdot)+5) = 0.5255;
                (*x_init)(0, ((2*i)+1)*(nQ+nQdot)+6) = -0.7440;
                (*x_init)(0, ((2*i)+1)*(nQ+nQdot)+7) = 0.6530;

                // BowFrog on ViolinBridge
                (*x_init)(1, ((2*i)+1)*(nQ+nQdot)+0) = 0.0611;
                (*x_init)(1, ((2*i)+1)*(nQ+nQdot)+1) = -0.5424;
                (*x_init)(1, ((2*i)+1)*(nQ+nQdot)+2) = 1.2640;
                (*x_init)(1, ((2*i)+1)*(nQ+nQdot)+3) = 0.5149;
                (*x_init)(1, ((2*i)+1)*(nQ+nQdot)+4) = 1.3547;
                (*x_init)(1, ((2*i)+1)*(nQ+nQdot)+5) = 0.7531;
                (*x_init)(1, ((2*i)+1)*(nQ+nQdot)+6) = -1.7864;
                (*x_init)(1, ((2*i)+1)*(nQ+nQdot)+7) = -0.4478;
            }
            break;
         }

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
