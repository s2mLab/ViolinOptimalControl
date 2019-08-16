#include <acado_optimal_control.hpp>
#include <bindings/acado_gnuplot/gnuplot_window.hpp>
#include "includes/dynamics.h"
#include "includes/objectives.h"
#include "includes/constraints.h"
#include <vector>
#include <time.h>

#ifndef PI
#define PI 3.141592
#endif

using namespace std;
USING_NAMESPACE_ACADO

/* ---------- Model ---------- */

biorbd::Model m("../../models/Bras.bioMod");

unsigned int nQ(m.nbQ());               // states number
unsigned int nQdot(m.nbQdot());         // derived states number
unsigned int nTau(m.nbGeneralizedTorque());           // torque number
unsigned int nTags(m.nTags());          // markers number
unsigned int nMus(m.nbMuscleTotal());   // muscles number
unsigned int nPhases(1);
GeneralizedCoordinates Q(nQ), Qdot(nQdot), Qddot(nQdot);
GeneralizedTorque Tau(nTau);
std::vector<biorbd::muscles::StateDynamics> state(nMus); // controls

const double t_Start=0.0;
const double t_End= 0.5;
const int nPoints(30);

int  main ()
{
    clock_t start,end;
    double time_exec;
    start=clock();

    std::cout << "nb de muscles: " << nMus << std::endl;
    std::cout << "nb de degré de liberté: " << nQ << std::endl;
    std::cout << "nb de torques: " << nTau << std::endl;

    /* ---------- INITIALIZATION ---------- */
    //Parameter               T;                              //  the  time  horizon T
    DifferentialState       x("",nQ+nQdot,1);               //  the  differential states
    Control                 u("", nMus+nTau, 1);                 //  the  control input  u
    IntermediateState       is(nQ + nQdot + nMus +nTau); // +1);
    TIME t;

    for (unsigned int i = 0; i < nQ+nQdot; ++i)
        is(i) = x(i);
    for (unsigned int i = 0; i < nMus+nTau; ++i)
        is(i+nQ+nQdot) = u(i);
   // is(nQ+nQdot+nMus+nTau) = T;


    /* ----------- DEFINE OCP ------------- */
    OCP ocp( t_Start, t_End , nPoints);

    CFunction mayer( 1, mayerVelocity);
    CFunction lagrangeRT( 1, lagrangeResidualTorques);
    CFunction lagrangeAcc(1, lagrangeAccelerations);
    CFunction lagrangeT(1, lagrangeTime);
    ocp.minimizeMayerTerm( mayer(is) );
    ocp.minimizeLagrangeTerm( lagrangeRT(is));

    /* ------------ CONSTRAINTS ----------- */
    DifferentialEquation    f(t_Start, t_End) ;
    CFunction F( nQ+nQdot, forwardDynamicsFromMuscleActivationAndTorque);
    CFunction frog( 4, violonUp);
    CFunction velocity(nQdot, velocityZero);
    CFunction tip( 4, violonDown);

    ocp.subjectTo( (f << dot(x)) == F(is)); //*T);                          //  differential  equation,
    ocp.subjectTo( AT_END, tip(is) ==  0.0 );
    ocp.subjectTo( AT_START, velocity(is) ==  0.0 );
    ocp.subjectTo( AT_START, frog(is) ==  0.0 );
    //ocp.subjectTo(0.1 <= T <= 4.0);

    for (unsigned int i=0; i<nMus; ++i){
         ocp.subjectTo(0.01 <= u(i) <= 1);
    }

    for (unsigned int i=nMus; i<nMus+nTau; ++i){
         ocp.subjectTo(-100 <= u(i) <= 100);
    }

    ocp.subjectTo(-PI/8 <= x(0) <= 0.1);
    ocp.subjectTo(-PI/2 <= x(1) <= 0.1);
    ocp.subjectTo(-PI/4 <= x(2) <= PI);
    ocp.subjectTo(-PI/2 <= x(3) <= PI/2);
    ocp.subjectTo(-0.1 <= x(4) <= PI);

    /* ---------- OPTIMIZATION  ------------ */
    OptimizationAlgorithm  algorithm( ocp ) ;       //  construct optimization  algorithm ,
    algorithm.set(MAX_NUM_ITERATIONS, 1000);
    algorithm.set(INTEGRATOR_TYPE, INT_RK45);
    algorithm.set(HESSIAN_APPROXIMATION, FULL_BFGS_UPDATE);
    algorithm.set(KKT_TOLERANCE, 1e-4);

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

// Tip -> Frog (montée)
//    x_init(0, 0) = 0.01;
//    x_init(0, 1) = -0.70;
//    x_init(0, 2) = 0.17;
//    x_init(0, 3) = 0.01;
//    x_init(0, 4) = 0.61;

//    x_init(1, 0) = 0.01;
//    x_init(1, 1) = -1.13;
//    x_init(1, 2) = 0.61;
//    x_init(1, 3) = -0.35;
//    x_init(1, 4) = 1.55;

//Frog -> Tip (descente)
    x_init(0, 0) = 0.01;
    x_init(0, 1) = -1.13;
    x_init(0, 2) = 0.61;
    x_init(0, 3) = -0.35;
    x_init(0, 4) = 1.55;

    x_init(1, 0) = 0.01;
    x_init(1, 1) = -0.70;
    x_init(1, 2) = 0.17;
    x_init(1, 3) = 0.01;
    x_init(1, 4) = 0.61;



    for(unsigned int i=nQ; i<nQ+nQdot; ++i){
         x_init(0, i) = 0.01;
         x_init(1, i) = 0.01;
    }

    algorithm.initializeDifferentialStates(x_init);

//    algorithm.initializeDifferentialStates("../Results/StatesAv2Musclesinit.txt");
//    algorithm.initializeControls("../Results/ControlsAv2Musclesinit.txt");
//    algorithm.set(PRINT_INTEGRATOR_PROFILE, BT_TRUE);

    GnuplotWindow window;                           //  visualize  the  results  in  a  Gnuplot  window
    window.addSubplot(  x ,  "STATES x" ) ;
    window.addSubplot( u ,  "CONTROL  u" ) ;
    algorithm << window;
    algorithm.solve();                              //  solve the problem

    algorithm.getDifferentialStates("../Results/StatesViolon.txt");
    //algorithm.getParameters("../Results/ParametersViolon.txt");
    algorithm.getControls("../Results/ControlsViolon.txt");

    end=clock();
    time_exec = double(end - start)/CLOCKS_PER_SEC;
    std::cout<<"Execution time: "<<time_exec<<std::endl;

    return 0;
}
