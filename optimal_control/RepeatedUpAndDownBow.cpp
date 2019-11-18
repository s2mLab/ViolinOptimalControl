#include <memory>
#include <time.h>
#include <acado_optimal_control.hpp>
#include <bindings/acado_gnuplot/gnuplot_window.hpp>

#include "biorbd.h"
#include "includes/utils.h"
#include "includes/dynamics.h"
#include "includes/constraints.h"
#include "includes/objectives.h"

// #define USE_INIT_FILE
biorbd::Model m("../../models/BrasViolon.bioMod");
#include "includes/biorbd_initializer.h"

static int idxSegmentBow = 8;
static int tagBowFrog(16);
static int tagBowTip(18);
static int tagViolinBridge(34);

const double t_Start = 0.0;
const double t_End = 0.5;
const int nPoints(31);
const int nPhases(2);
const std::string resultsPath("../Results/");
const std::string initializePath("../Initialisation/");
const std::string optimizationName("RepeatedUpAndDownBow");

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
    CFunction lagrangeRT(1, lagrangeResidualTorques);
    CFunction lagrangeA(1, lagrangeActivations);
    CFunction F( nQ+nQdot, forwardDynamics_noContact);
    DifferentialEquation f ;

    // --------- DEFINE SOME PATH CONSTRAINTS --------- //
    CFunction markerBowFrog(3, markerPosition);
    markerBowFrog.setUserData(static_cast<void*>(&tagBowFrog));
    CFunction markerBowTip(3, markerPosition);
    markerBowTip.setUserData(static_cast<void*>(&tagBowTip));
    CFunction markerViolinBridge(3, markerPosition);
    markerViolinBridge.setUserData(static_cast<void*>(&tagViolinBridge));
    CFunction violinBridgeInBowRT(2, projectOnXyPlane);
    int idxProjectViolinBridgeInBow[2] = {tagViolinBridge, idxSegmentBow};
    violinBridgeInBowRT.setUserData(static_cast<void*>(idxProjectViolinBridgeInBow));


    // ---------- INITIALIZATION ---------- //
    std::vector<DifferentialState> x;
    std::vector<Control> control;
    std::vector<IntermediateState> is;

    // ---------- PHASES ---------- //
    for (unsigned int p=0; p<nPhases; ++p){
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
                        AT_START, markerBowFrog(x[p]) - markerViolinBridge(x[p])
                        == 0.0 );
            ocp.subjectTo(
                        AT_END, markerBowTip(x[p]) - markerViolinBridge(x[p])
                        == 0.0 );
        }
        else {
            ocp.subjectTo( 0.0, x[p], -x[p-1], 0.0 );
            ocp.subjectTo( 0.0, x[p-1], -x[p], 0.0 );
        }
        for (int i = 1; i < nPoints-1; ++i) {
            ocp.subjectTo(i, violinBridgeInBowRT(x[p]) == 0.0);
        }

        ocp.subjectTo(-PI/8 <= x[p](0) <= 0.1);
        ocp.subjectTo(-PI/2 <= x[p](1) <= 0.1);
        ocp.subjectTo(-PI/4 <= x[p](2) <= PI);
        ocp.subjectTo(-PI/2 <= x[p](3) <= PI/2);
        ocp.subjectTo(-0.1  <= x[p](4) <= PI);
        ocp.subjectTo(-PI   <= x[p](5) <= PI);
        ocp.subjectTo(-PI   <= x[p](6) <= PI);

        for (unsigned int j=0; j<nQdot; ++j) {
            ocp.subjectTo(-50 <= x[p](nQ + j) <= 50);
        }

    }
    ocp.subjectTo(f);

    // ------------ OBJECTIVE ----------- //
    Expression sumLagrange = lagrangeRT(control[0])+ lagrangeA(control[0]);
    for(unsigned int p=1; p<nPhases; ++p)
        sumLagrange += lagrangeRT(control[p]) + lagrangeA(control[p]);
    ocp.minimizeLagrangeTerm( sumLagrange ); // WARNING

    // ---------- OPTIMIZATION  ------------ //
    OptimizationAlgorithm  algorithm(ocp) ;
    algorithm.set(MAX_NUM_ITERATIONS, 500);
    algorithm.set(INTEGRATOR_TYPE, INT_RK45);
    algorithm.set(HESSIAN_APPROXIMATION, FULL_BFGS_UPDATE);
    algorithm.set(KKT_TOLERANCE, 1e-4);


    // ---------- INITIAL SOLUTION ---------- //
#ifdef USE_INIT_FILE
    algorithm.initializeDifferentialStates(
                (initializePath + "InitStates" + resultsName + ".txt").c_str(),
                BT_TRUE);
    algorithm.initializeControls(
                (initializePath + "InitControls" + resultsName + ".txt").c_str()
                );
#else
    VariablesGrid u_init(nPhases*(nTau + nMus), Grid(t_Start, t_End, 2));
    for(unsigned int i=0; i<nPhases; ++i){
        for(unsigned int j=0; j<nMus; ++j){
            u_init(0, i*(nMus+nTau) + j ) = 0.2;
            u_init(1, i*(nMus+nTau) + j ) = 0.2;
        }
        for(unsigned int j=0; j<nTau; ++j){
            u_init(0, i*(nMus+nTau) + nMus + j ) = 0.01;
            u_init(1, i*(nMus+nTau) + nMus + j ) = 0.01;
        }
    }
    algorithm.initializeControls(u_init);

    VariablesGrid x_init(nPhases*(nQ+nQdot), Grid(t_Start, t_End, 2));
    for(unsigned int i=0; i < nPhases/2; ++i){
        // BowFrog on ViolinBridge
        x_init(0, 2*i*(nQ+nQdot)+0) = 0.09973;
        x_init(0, 2*i*(nQ+nQdot)+1) = 0.09733;
        x_init(0, 2*i*(nQ+nQdot)+2) = 1.05710;
        x_init(0, 2*i*(nQ+nQdot)+3) = 1.56950;
        x_init(0, 2*i*(nQ+nQdot)+4) = 1.07125;
        x_init(0, 2*i*(nQ+nQdot)+5) = 0.95871;
        x_init(0, 2*i*(nQ+nQdot)+6) = -1.7687;

        // BowTip on ViolinBridge
        x_init(1, 2*i*(nQ+nQdot)+0) = -0.39107;
        x_init(1, 2*i*(nQ+nQdot)+1) = -0.495383;
        x_init(1, 2*i*(nQ+nQdot)+2) = -0.089030;
        x_init(1, 2*i*(nQ+nQdot)+3) = 0.1485315;
        x_init(1, 2*i*(nQ+nQdot)+4) = 0.8569764;
        x_init(1, 2*i*(nQ+nQdot)+5) = 1.9126840;
        x_init(1, 2*i*(nQ+nQdot)+6) = -0.490220;

        // BowTip on ViolinBridge
        x_init(0, ((2*i)+1)*(nQ+nQdot)+0) = -0.39107;
        x_init(0, ((2*i)+1)*(nQ+nQdot)+1) = -0.495383;
        x_init(0, ((2*i)+1)*(nQ+nQdot)+2) = -0.089030;
        x_init(0, ((2*i)+1)*(nQ+nQdot)+3) = 0.1485315;
        x_init(0, ((2*i)+1)*(nQ+nQdot)+4) = 0.8569764;
        x_init(0, ((2*i)+1)*(nQ+nQdot)+5) = 1.9126840;
        x_init(0, ((2*i)+1)*(nQ+nQdot)+6) = -0.490220;

        // BowFrog on ViolinBridge
        x_init(1, ((2*i)+1)*(nQ+nQdot)+0) = 0.09973;
        x_init(1, ((2*i)+1)*(nQ+nQdot)+1) = 0.09733;
        x_init(1, ((2*i)+1)*(nQ+nQdot)+2) = 1.05710;
        x_init(1, ((2*i)+1)*(nQ+nQdot)+3) = 1.56950;
        x_init(1, ((2*i)+1)*(nQ+nQdot)+4) = 1.07125;
        x_init(1, ((2*i)+1)*(nQ+nQdot)+5) = 0.95871;
        x_init(1, ((2*i)+1)*(nQ+nQdot)+6) = -1.7687;
    }
    for(unsigned int i=0; i<nPhases; ++i){
        for(unsigned int j=0; j<nQdot; ++j){
             x_init(0, i*(nQ+nQdot) + nQ + j) = 0.01;
             x_init(1, i*(nQ+nQdot) + nQ + j) = 0.01;
        }
    }
    algorithm.initializeDifferentialStates(x_init);
#endif  // USE_INIT_FILE

    // ---------- SOLVING THE PROBLEM ---------- //
    algorithm.solve();

    // ---------- STORING THE RESULTS ---------- //
   createTreePath(resultsPath);
   algorithm.getDifferentialStates((resultsPath + "States" + optimizationName + ".txt").c_str());
   algorithm.getControls((resultsPath + "Controls" + optimizationName + ".txt").c_str());

    // ---------- STORING THE RESULTS FOR DUPLICATION ---------- //
    VariablesGrid states, controls, doubleState,
            doubleState2, doubleControls, doubleControls2;

    algorithm.getDifferentialStates(states); // On stocke les états différentiels de la simulation en cours
    algorithm.getControls(controls); // On stocke les contrôles de la simulation en cours

    doubleState = states.getValuesSubGrid(0, ((nQ + nQdot)* nPhases) - 1 ); // On récupere les valeurs des états différentiels sans la derniere colonne
    doubleState2 = states.getValuesSubGrid(0, (nQ + nQdot) * nPhases); // On récupere les valeurs des états différentiels avec la derniere colonne
    doubleState.appendValues(doubleState2); // On rejoins les deux VariablesGrid dans l'ordre "doubleStates + doubleStates2"
    doubleState.print((resultsPath + "InitStates" + optimizationName + ".txt").c_str()); // On remplit le fichier texte avec le VariablesGrid final

    doubleControls = controls.getValuesSubGrid(0, (nTau + nMus) * nPhases - 1); // Pareil avec les contrôles
    doubleControls2 = controls.getValuesSubGrid(0, (nTau + nMus) * nPhases - 1);
    doubleControls.appendValues(doubleControls2);
    doubleControls.print((resultsPath + "InitControls" + optimizationName + ".txt").c_str());

    // ---------- PLOTING ---------- //
    clock_t end=clock();
    double time_exec(double(end - start)/CLOCKS_PER_SEC);
    std::cout<<"Execution time: "<<time_exec<<std::endl;

    // ---------- EXIT ---------- //
    return 0;
}
