#include <acado_optimal_control.hpp>
#include <bindings/acado_gnuplot/gnuplot_window.hpp>
#include "includes/dynamics.h"
#include "includes/objectives.h"
#include "includes/constraints.h"
#include <vector>

using namespace std;
USING_NAMESPACE_ACADO

/* ---------- Model ---------- */

s2mMusculoSkeletalModel m("../../models/Bras.bioMod");

unsigned int nQ(m.nbQ());               // states number
unsigned int nQdot(m.nbQdot());         // derived states number
unsigned int nTau(m.nbTau());           // torque number
unsigned int nTags(m.nTags());          // markers number
unsigned int nMus(m.nbMuscleTotal());   // muscles number

const double t_Start=0.0;
const double t_End= 1.0;
const int nPoints(30);

int  main ()
{
    std::cout << "nb de muscles: " << nMus << std::endl;
    std::cout << "nb de degré de liberté: " << nQ << std::endl;
    std::cout << "nb de torques: " << nTau << std::endl;

    /* ---------- INITIALIZATION ---------- */
    Parameter               T;                              //  the  time  horizon T
    DifferentialState       x("",nQ+nQdot,1);               //  the  differential states
    Control                 u("", nMus+nTau, 1);                 //  the  control input  u
    IntermediateState       is(nQ + nQdot + nMus +nTau);


    for (unsigned int i = 0; i < nQ+nQdot; ++i)
        is(i) = x(i);
    for (unsigned int i = 0; i < nMus+nTau; ++i)
        is(i+nQ+nQdot) = u(i);


    /* ----------- DEFINE OCP ------------- */
    OCP ocp( t_Start, t_End , nPoints);

    CFunction Mayer( 1, MayerSpeed);
    CFunction Lagrange( 1, LagrangeAddedTorques);
    ocp.minimizeMayerTerm( Mayer(is) );
    ocp.minimizeLagrangeTerm( Lagrange(is) );

    /* ------------ CONSTRAINTS ----------- */
    DifferentialEquation    f ;
    CFunction F( nQ+nQdot, forwardDynamicsFromMuscleActivationAndTorque);
    CFunction I( nQ+nQdot, StatesZero);
    CFunction E( 1, Rotbras);

    ocp.subjectTo( (f << dot(x)) == F(is)*T );                          //  differential  equation,
    ocp.subjectTo( AT_START, I(is) ==  0.0 );
    ocp.subjectTo( AT_END  , E(is) ==  0.0 );
    ocp.subjectTo(0.1 <= T <= 4.0);

    for (unsigned int i=0; i<nMus; ++i){
         ocp.subjectTo(0.01 <= u(i) <= 1);
    }

    for (unsigned int i=nMus; i<nMus+nTau; ++i){
         ocp.subjectTo(-20 <= u(i) <= 20);
    }

    ocp.subjectTo(-PI/8 <= x(0) <= 0.1);
    ocp.subjectTo(-PI/2 <= x(1) <= 0.1);
    ocp.subjectTo(-PI/4 <= x(2) <= PI);
    ocp.subjectTo(-PI/2 <= x(3) <= PI/2);
    ocp.subjectTo(-0.1 <= x(4) <= PI);

    /* ---------- OPTIMIZATION  ------------ */
    OptimizationAlgorithm  algorithm( ocp ) ;       //  construct optimization  algorithm ,
    algorithm.set(MAX_NUM_ITERATIONS, 1000);
    //algorithm.set(KKT_TOLERANCE, 1e-10);
    //algorithm.set(INTEGRATOR_TOLERANCE, 1e-6);

    VariablesGrid u_init(nTau + nMus, Grid(t_Start, t_End, 2));
    for(unsigned int i=0; i<2; ++i){
        for(unsigned int j=0; j<nMus; ++j){
            u_init(i, j) = 0.02;
        }
        for(unsigned int j=nMus; j<nMus+nTau; ++j){
            u_init(i, j) = 0.001;
        }
    }
    algorithm.initializeControls(u_init);

    VariablesGrid x_init(nQ+nQdot, Grid(t_Start, t_End, 2));
    for(unsigned int i=0; i<nQ-1; ++i){
         x_init(0, i) = 0.01;
         x_init(1, i) = 0.01;
    }

    x_init(0, 4) = 0.01;
    x_init(1, 4) = 0.8;

    for(unsigned int i=nQ; i<nQdot; ++i){
         x_init(0, i) = 0.01;
         x_init(1, i) = 0.01;
    }
    algorithm.initializeDifferentialStates(x_init);


    GnuplotWindow window;                           //  visualize  the  results  in  a  Gnuplot  window
    window.addSubplot(  x ,  "STATES x" ) ;
    window.addSubplot( u ,  "CONTROL  u" ) ;
    window.addSubplot( T ,  "Time " ) ;
    algorithm << window;
    algorithm.solve();                              //  solve the problem

    algorithm.getDifferentialStates("../Results/StatesAv2Muscles.txt");
    algorithm.getParameters("../Results/ParametersAv2Muscles.txt");
    algorithm.getControls("../Results/ControlsAv2Muscles.txt");

    return 0;
}
