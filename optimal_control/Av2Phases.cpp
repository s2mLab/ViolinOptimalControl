#include <acado_optimal_control.hpp>
#include <bindings/acado_gnuplot/gnuplot_window.hpp>
#include "includes/dynamics.h"

using namespace std;
USING_NAMESPACE_ACADO

/* ---------- Model ---------- */

s2mMusculoSkeletalModel m("../Modeles/ModeleAv1Muscle.bioMod");
unsigned int nQ(m.nbQ());               // states number
unsigned int nQdot(m.nbQdot());         // derived states number
unsigned int nTau(m.nbTau());           // controls number
unsigned int nTags(m.nTags());          // markers number
unsigned int nMus(m.nbMuscleTotal());   // muscles number

const double t_Start = 0.0;
const double t_End = 10.0;
const int nPoints(30);

/* ---------- Functions ---------- */

#define  NX   nQ + nQdot        // number of differential states

#define  NOL   1                 // number of lagrange objective functions
void myLagrangeObjectiveFunction( double *x, double *g, void * ){
    g[0] = x[2];
}


#define  NOM   1                 // number of mayer objective functions
void myMayerObjectiveFunction( double *x, double *g, void * ){
    g[0] = x[1];
}

#define  NI   2                 // number of initial value constraints
void myInitialValueConstraint( double *x, double *g, void * ){
    g[0] = x[0]-0.01;
    g[1] = x[1];

}

#define  NE1   2                 // number of end-point / terminal constraints
void myEndPointConstraint1( double *x, double *g, void * ){
    g[0]=x[0]-PI/4;                         // rotation de 90°
    g[1]=x[1];                              // vitesse nulle

}

#define  NE2   1                 // number of end-point / terminal constraints
void myEndPointConstraint2( double *x, double *g, void * ){
    g[0]=x[0]-PI/2;                         // rotation de 90°

}

int  main ()
{
    std::cout << "nb de marqueurs: " << nTags << std::endl<< std::endl;
    std::cout << "nb de muscles: " << nMus << std::endl<< std::endl;

    /* ---------- INITIALIZATION ---------- */
    DifferentialState       x1("",nQ+nQdot,1);               //  the  differential states
    DifferentialState       x2("",nQ+nQdot,1);
    Control                 u1("", nMus, 1);                 //  the  control input  u
    Control                 u2("", nMus, 1);
    Parameter               T1;
    Parameter               T2;

    IntermediateState       is1(nQ + nQdot + nMus);
    IntermediateState       is2(nQ + nQdot + nMus);

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

    /* ----------- DEFINE OCP ------------- */
    OCP ocp(0.0, 1.0, nPoints);

    CFunction Mayer( NOM, myMayerObjectiveFunction);
    CFunction Lagrange( NOL, myLagrangeObjectiveFunction);
    ocp.minimizeMayerTerm( Mayer(is2) );
    ocp.minimizeLagrangeTerm( Lagrange(is1) );

    /* ------------ CONSTRAINTS ----------- */
    CFunction F( NX, forwardDynamicsFromMuscleActivation);

    DifferentialEquation    f ;
    (f << dot(x1)) == F(is1)/T1;
    (f << dot(x2)) == F(is2)/T2;
    ocp.subjectTo(f);

    CFunction I( NI, myInitialValueConstraint   );
    CFunction E1( NE1, myEndPointConstraint1       );
    CFunction E2( NE2, myEndPointConstraint2       );

    ocp.subjectTo( AT_START, I(x1) ==  0.0 );
    ocp.subjectTo( 0.0, x2, -x1, 0.0 );
    ocp.subjectTo( AT_END  , E1(x1) ==  0.0 );
    ocp.subjectTo(AT_END, E2(x2) == 0.0);

    ocp.subjectTo(0.01 <= u1 <= 1);
    ocp.subjectTo(0.01 <= u2 <= 1);

    ocp.subjectTo(0.1 <= T1 <= 5.0);
    ocp.subjectTo(0.1 <= T2 <= 5.0);

    /* ---------- OPTIMIZATION  ------------ */
    OptimizationAlgorithm  algorithm( ocp ) ;       //  construct optimization  algorithm ,

    algorithm.initializeDifferentialStates("../Initialisation/X2Phases.txt");
    algorithm.initializeControls("../Initialisation/U2Phases.txt");
    algorithm.initializeParameters("../Initialisation/T2Phases.txt");

    GnuplotWindow window;                           //  visualize  the  results  in  a  Gnuplot  window
    window.addSubplot(  x1 ,  "STATES x" ) ;
    window.addSubplot(  x2 ,  "STATES x" ) ;
    window.addSubplot( u1 ,  "CONTROL  u" ) ;
    window.addSubplot( u2 ,  "CONTROL  u" ) ;
    algorithm << window;
    algorithm.solve();                              //  solve the problem .

    algorithm.getDifferentialStates("../Results/StatesAv2Phases.txt");
    //algorithm.getParameters("../Results/ParametersAv2Phases.txt");
    algorithm.getControls("../Results/ControlsAv2Phases.txt");

    return 0;
}

