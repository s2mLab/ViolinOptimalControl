#include <acado_optimal_control.hpp>
#include <bindings/acado_gnuplot/gnuplot_window.hpp>
#include <vector>
#include <memory>

#include "biorbd.h"
#include "includes/dynamics.h"
biorbd::Model m("../../models/simple.bioMod");
#include "includes/biorbd_initializer.h"

const std::string optimizationName("eocarBiorbd");
const std::string resultsPath("../Results/");
const std::string controlResultsFileName(resultsPath + "Controls" + optimizationName + ".txt");
const std::string stateResultsFileName(resultsPath + "States" + optimizationName + ".txt");

USING_NAMESPACE_ACADO

// ---------- Model ---------- //

const double t_Start = 0.0;
const double t_End = 2.0;
const int nPoints(30);

int  main(){
    CFunction cDynamics(m.nbQ() + m.nbQdot(), forwardDynamics_noContact);

    // ----------- DEFINE OCP ------------- //
    OCP ocp(t_Start, t_End, nPoints);

    // ------------ CONSTRAINTS ----------- //
    DifferentialState x("",nQ + nQdot,1);
    Control u("", nTau, 1);
    IntermediateState is("", nTau + nQ + nQdot, 1);

    for (unsigned int i = 0; i < nQ + nQdot; ++i)
        is(i) = x(i);
    for (unsigned int i = 0; i < nTau; ++i)
        is(i+nQ + nQdot) = u(i);

    DifferentialEquation f;
    ocp.subjectTo( (f << dot(x)) == cDynamics(is) );

    for (int i=0; i<m.nbQ(); ++i){
        ocp.subjectTo( AT_START, x(i) ==  0 );
        if (i == 0){
            ocp.subjectTo( AT_END, x(0) == 10.0);
        } else if (i == 1) {
            ocp.subjectTo( AT_END, x(1) == 10.0);
        } else if (i == 2) {
            ocp.subjectTo( AT_END, x(2) == 0.0);
        } else if (i == 3) {
            ocp.subjectTo( AT_END, x(3) == M_PI/4);
        } else {
            throw std::runtime_error("Not valid dof");
        }
    }
    for (int i=m.nbQ(); i<m.nbQ() + m.nbQdot(); ++i){
        ocp.subjectTo( AT_START, x(i) == 0);
        ocp.subjectTo( AT_END, x(i) == 0);
    }
    for (unsigned int i=0; i<nTau; ++i)
        ocp.subjectTo(-100 <= u(i) <= 100);
    for (unsigned int i=0; i<nQ+nQdot; ++i)
        ocp.subjectTo(-100 <= x(i) <= 100);


    // ------------ OBJECTIVE ----------- //
    Expression sumLagrange(u*u);
    ocp.minimizeLagrangeTerm(sumLagrange);

    // ---------- VISUALIZATION ------------ //
    GnuplotWindow window;
    for (int i=0; i<m.nbQ();  ++i){
        window.addSubplot( x(i), "Position" );
    }
    for (int i=m.nbQ(); i<m.nbQ()+m.nbQdot();  ++i){
        window.addSubplot( x(i), "Vitesse" );
    }
    for (int i=0; i<m.nbGeneralizedTorque();  ++i){
        window.addSubplot( u(i),  "Acceleration" );
    }


    // ---------- OPTIMIZATION  ------------ //
    OptimizationAlgorithm  algorithm( ocp ) ;
    algorithm.set(INTEGRATOR_TYPE, INT_RK45);
    algorithm.set(HESSIAN_APPROXIMATION, FULL_BFGS_UPDATE);
    algorithm.set(KKT_TOLERANCE, 1e-6);
    algorithm << window;
    clock_t start = clock();
    algorithm.solve();
    clock_t end=clock();

    // ---------- SHOW SOLUTION  ------------ //
    VariablesGrid finalU, finalX;
    algorithm.getControls(finalU);
    algorithm.getDifferentialStates(finalX);
    algorithm.set(KKT_TOLERANCE, 1e-6);

    createTreePath(resultsPath);
    algorithm.getDifferentialStates(stateResultsFileName.c_str());
    algorithm.getControls(controlResultsFileName.c_str());
//    finalU.print();
//    finalX.print();


    // ---------- FINALIZE  ------------ //
    double time_exec(double(end - start)/CLOCKS_PER_SEC);
    std::cout<<"Execution time: "<<time_exec<<std::endl;
    return  0;
}
