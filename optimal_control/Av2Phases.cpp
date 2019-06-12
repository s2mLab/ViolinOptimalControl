#include <acado_optimal_control.hpp>
#include <bindings/acado_gnuplot/gnuplot_window.hpp>
#include "includes/dynamics.h"
#include "includes/objectives.h"
#include "includes/constraints.h"

using namespace std;
USING_NAMESPACE_ACADO

/* ---------- Model ---------- */

s2mMusculoSkeletalModel m(".../../models/ModeleAv1Muscle.bioMod");
unsigned int nQ(m.nbQ());               // states number
unsigned int nQdot(m.nbQdot());         // derived states number
unsigned int nTau(m.nbTau());           // controls number
unsigned int nTags(m.nTags());          // markers number
unsigned int nMus(m.nbMuscleTotal());   // muscles number

const double t_Start = 0.0;
const double t_End = 10.0;
const int nPoints(30);

/* ---------- Functions ---------- */

int  main ()
{
    std::cout << "nb de marqueurs: " << nTags << std::endl;
    std::cout << "nb de muscles: " << nMus << std::endl;
    std::cout << "nb de torques: " << nTau << std::endl;

    /* ---------- INITIALIZATION ---------- */
    DifferentialState       x1("",nQ+nQdot,1);               //  the  differential states
    DifferentialState       x2("",nQ+nQdot,1);
    Control                 u1("", nMus+nTau, 1);                 //  the  control input  u
    Control                 u2("", nMus+nTau, 1);
    Parameter               T1;
    Parameter               T2;

    IntermediateState       is1(nQ + nQdot + nMus + nTau + 1);
    IntermediateState       is2(nQ + nQdot + nMus + nTau + 1);

    for (unsigned int i = 0; i < nQ; ++i){
        is1(i) = x1(i);
        is2(i) = x2(i);
    }
    for (unsigned int i = 0; i < nQdot; ++i){
        is1(i+nQ) = x1(i+nQ);
        is2(i+nQ) = x2(i+nQ);
    }
    for (unsigned int i = 0; i < nMus; ++i){
        is1(i+nQ+nQdot) = u1(i);
        is2(i+nQ+nQdot) = u2(i);
    }
    for (unsigned int i = 0; i < nTau; ++i){
        is1(i+nQ+nQdot+nMus) = u1(i);
        is2(i+nQ+nQdot+nMus) = u2(i);
    }
    is1(nQ+nQdot+nMus+nTau)=T1;
    is2(nQ+nQdot+nMus+nTAu)=T2;

    /* ----------- DEFINE OCP ------------- */
    OCP ocp(0.0, 1.0, nPoints);

    CFunction Mayer(1, MayerVelocity);
    CFunction Lagrange(1, LagrangeResidualTorques);
    ocp.minimizeMayerTerm( Mayer(is1) + Mayer(is2));
    ocp.minimizeLagrangeTerm( Lagrange(is1) + Lagrange(is2) );

    /* ------------ CONSTRAINTS ----------- */
    CFunction F( NX, forwardDynamicsFromMuscleActivation);

    DifferentialEquation    f ;
    (f << dot(x1)) == F(is1)*T1;
    (f << dot(x2)) == F(is2)*T2;
    ocp.subjectTo(f);

    CFunction I( 3, ViolonUp);
    CFunction E1( 3,  ViolonDown);
    CFunction E2( 3, ViolonUp);

    ocp.subjectTo( AT_START, I(x1) ==  0.0 );
    ocp.subjectTo( 0.0, x2, -x1, 0.0 );
    ocp.subjectTo( AT_END  , E1(x1) ==  0.0 );
    ocp.subjectTo(AT_END, E2(x2) == 0.0);

    for (unsigned int i=0; i<nMus; ++i){
         ocp.subjectTo(0.01 <= u1(i) <= 1);
         ocp.subjectTo(0.01 <= u2(i) <= 1);
    }

    for (unsigned int i=nMus; i<nMus+nTau; ++i){
         ocp.subjectTo(-100 <= u1(i) <= 100);
         ocp.subjectTo(-100 <= u2(i) <= 100);
    }

    ocp.subjectTo(0.1 <= T1 <= 5.0);
    ocp.subjectTo(0.1 <= T2 <= 5.0);

    ocp.subjectTo(-PI/8 <= x1(0) <= 0.1);
    ocp.subjectTo(-PI/2 <= x1(1) <= 0.1);
    ocp.subjectTo(-PI/4 <= x1(2) <= PI);
    ocp.subjectTo(-PI/2 <= x1(3) <= PI/2);
    ocp.subjectTo(-0.1 <= x1(4) <= PI);

    ocp.subjectTo(-PI/8 <= x2(0) <= 0.1);
    ocp.subjectTo(-PI/2 <= x2(1) <= 0.1);
    ocp.subjectTo(-PI/4 <= x2(2) <= PI);
    ocp.subjectTo(-PI/2 <= x2(3) <= PI/2);
    ocp.subjectTo(-0.1 <= x2(4) <= PI);

    /* ---------- OPTIMIZATION  ------------ */
    OptimizationAlgorithm  algorithm( ocp ) ;       //  construct optimization  algorithm ,
    algorithm.set(MAX_NUM_ITERATIONS, 1000);
    algorithm.set(INTEGRATOR_TYPE, INT_RK45);
    algorithm.set(HESSIAN_APPROXIMATION, FULL_BFGS_UPDATE);
    algorithm.set(KKT_TOLERANCE, 1e-6);

    VariablesGrid u_init(2*(nTau + nMus), Grid(t_Start, t_End, 2));
    for(unsigned int i=0; i<2; ++i){
        for(unsigned int j=0; j<nMus; ++j){
            u_init(i, j) = 0.02;
        }
        for(unsigned int j=nMus; j<nMus+nTau; ++j){
            u_init(i, j) = 0.001;
        }
        for(unsigned int j=nMus+nTau; j<2*nMus+nTau; ++j){
            u_init(i, j) = 0.02;
        }
        for(unsigned int j=2*nMus+nTau; j<2*(nMus+nTau); ++j){
            u_init(i, j) = 0.001;
        }
    }
    algorithm.initializeControls(u_init);

    VariablesGrid x_init(2*(nQ+nQdot), Grid(t_Start, t_End, 2));
    for(unsigned int i=0; i<nQ; ++i){
        x_init(0, i) = 0.01;
        x_init(1, i) = 0.01;
        x_init(0, i+nQ+nQdot) = 0.01;
        x_init(1, i+nQ+nQdot) = 0.01;
    }

    x_init(0, 1) = -1.31;
    x_init(0, 2) = 1.22;
    x_init(0, 4) = 1.92;

    x_init(1, 1) = -0.87;
    x_init(1, 4) = 0.17;

    x_init(0, 1+nQ+nQdot) = -0.87;
    x_init(0, 4+nQ+nQdot) = 0.17;

    x_init(1, 1+nQ+nQdot) = -1.31;
    x_init(1, 2+nQ+nQdot) = 1.22;
    x_init(1, 4+nQ+nQdot) = 1.92;


    for(unsigned int i=nQ; i<nQ+nQdot; ++i){
         x_init(0, i) = 0.01;
         x_init(1, i) = 0.01;
         x_init(0, i+nQ+nQdot) = 0.01;
         x_init(1, i+nQ+nQdot) = 0.01;

    }

    algorithm.initializeDifferentialStates(x_init);


    GnuplotWindow window;                           //  visualize  the  results  in  a  Gnuplot  window
    window.addSubplot(  x1 ,  "STATES x" ) ;
    window.addSubplot(  x2 ,  "STATES x" ) ;
    window.addSubplot( u1 ,  "CONTROL  u" ) ;
    window.addSubplot( u2 ,  "CONTROL  u" ) ;
    algorithm << window;
    algorithm.solve();                              //  solve the problem .

    algorithm.getDifferentialStates("../Results/StatesAv2Phases.txt");
    algorithm.getParameters("../Results/ParametersAv2Phases.txt");
    algorithm.getControls("../Results/ControlsAv2Phases.txt");

    return 0;
}

