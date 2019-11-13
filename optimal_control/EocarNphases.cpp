#include <acado_optimal_control.hpp>
#include <bindings/acado_gnuplot/gnuplot_window.hpp>
#include "BiorbdModel.h"
#include "includes/dynamics.h"
#include "includes/objectives.h"
#include "includes/constraints.h"
#include "includes/utils.h"
#include <vector>
#include <memory>

USING_NAMESPACE_ACADO

/* ---------- Model ---------- */
biorbd::Model m("../../models/ModelTest.bioMod");
#include "includes/biorbd_initializer.h"

const double t_Start = 0.0;
const double t_End = 4.0;
const int nPoints(31);
const int nPhases(2);
const std::string resultsPath("../Results/");
const std::string initializePath("../Initialisation/");
const std::string resultsName("Eocar");

int  main ()
{
    clock_t start = clock();
    std::cout << "nb de muscles: " << nMus << std::endl;
    std::cout << "nb de torques: " << nTau << std::endl;
    std::cout << "nb de marqueurs: " << nMarkers << std::endl;
    initializeMuscleStates();

    // ----------- DEFINE OCP ------------- //
    OCP ocp(t_Start, t_End, nPoints);
    CFunction F( nQ+nQdot, forwardDynamics_noContact);
    DifferentialEquation f ;


    // ---------- INITIALIZATION ---------- //
    std::vector<DifferentialState> x1;
    std::vector<Control> mus1;
    std::vector<Control> torque1;
    std::vector<IntermediateState> is1;

    // ------------ CONSTRAINTS ----------- //
    for (unsigned int p=0; p<nPhases; ++p){

        x1.push_back(DifferentialState("",nQ+nQdot,1));
        mus1.push_back(Control("",  nMus, 1));
        torque1.push_back(Control("", nTau, 1));
        is1.push_back(IntermediateState(nQ + nQdot + nMus + nTau));

        for (unsigned int i = 0; i < nQ; ++i)
            is1[p](i) = x1[p](i);
        for (unsigned int i = 0; i < nQdot; ++i)
            is1[p](i+nQ) = x1[p](i+nQ);
        for (unsigned int i = 0; i < nMus; ++i)
            is1[p](i+nQ+nQdot) = mus1[p](i);
        for (unsigned int i = 0; i < nTau; ++i)
            is1[p](i+nQ+nQdot+nMus) = torque1[p](i);

        (f << dot( x1[p] )) == F(is1[p]);

        if (p == 0) {
            ocp.subjectTo( AT_START, x1[p](0) ==  0 );
            ocp.subjectTo( AT_END, x1[p](0) ==  10 );

            ocp.subjectTo( AT_START, x1[p](1) ==  0);
            ocp.subjectTo( AT_END, x1[p](1) ==  0);
        }
        else {
            ocp.subjectTo( 0.0, x1[p], -x1[p-1], 0.0 );
            ocp.subjectTo( 0.0, x1[p-1], -x1[p], 0.0 );
        }
        //    ocp.subjectTo( AT_START, x1(2) ==  0 );
        //    ocp.subjectTo( AT_END, x1(2) ==  0 );

        //    ocp.subjectTo( AT_START, x1(3) ==  0 );
        //    ocp.subjectTo( AT_END, x1(3) ==  0 );

        for (unsigned int i=0; i<nMus; ++i)
            ocp.subjectTo(0.01 <= mus1[p](i) <= 1);
        for (unsigned int i=0; i<nTau; ++i)
            ocp.subjectTo(-100 <= torque1[p](i) <= 100);
   }
    ocp.subjectTo( f ) ;

    // ------------ OBJECTIVE ----------- //
    Expression sumLagrange(torque1[0] * torque1[0] + mus1[0]*mus1[0]);
    for(unsigned int p=1; p<nPhases; ++p) {
        sumLagrange += torque1[p] * torque1[p] + mus1[p]*mus1[p];
    }
    ocp.minimizeLagrangeTerm(sumLagrange);

    // ---------- VISUALIZATION ------------ //

    GnuplotWindow window;
    for (unsigned int p=0; p<nPhases; ++p){
        window.addSubplot( x1[p](0), "Etat 0 " );
        window.addSubplot( x1[p](1), "Etat 1 " );
        window.addSubplot( mus1[p](0),  "CONTROL  1" ) ;
        window.addSubplot( torque1[p](0),  "CONTROL  2" ) ;
    }


    /* ---------- OPTIMIZATION  ------------ */

    OptimizationAlgorithm  algorithm( ocp ) ;       //  construct optimization  algorithm ,
    algorithm.set(MAX_NUM_ITERATIONS, 500);
    algorithm.initializeDifferentialStates((resultsPath + "InitStates" + resultsName + ".txt").c_str(), BT_TRUE);
    algorithm.initializeControls((resultsPath + "InitControls" + resultsName + ".txt").c_str());

    algorithm << window;
    algorithm.solve();

    VariablesGrid controls, test;

    algorithm.getDifferentialStates(test);
    algorithm.getControls(controls);

    VariablesGrid n1 = test.getValuesSubGrid(0 , (nQ + nQdot) * (nPhases - 1) + 1);
    VariablesGrid n2 = test.getValuesSubGrid(0, (nQ + nQdot) * (nPhases - 1) + 2);

    VariablesGrid n3 = controls.getValuesSubGrid(0, nTau * (nPhases - 1));
    VariablesGrid n4 = controls.getValuesSubGrid(0, nTau * (nPhases - 1));

    n1.appendValues(n2);
    n1.print((resultsPath + "InitStates" + resultsName + ".txt").c_str());

    n3.appendValues(n4);
    n3.print((resultsPath + "InitControls" + resultsName + ".txt").c_str());

    createTreePath(resultsPath);
    algorithm.getDifferentialStates((resultsPath + "States" + resultsName + ".txt").c_str());
    algorithm.getControls((resultsPath + "Controls" + resultsName + ".txt").c_str());


    clock_t end=clock();
    double time_exec(double(end - start)/CLOCKS_PER_SEC);
    std::cout<<"Execution time: "<<time_exec<<std::endl;
    return  0;
}
