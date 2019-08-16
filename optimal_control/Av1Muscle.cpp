#include <acado_optimal_control.hpp>
#include <bindings/acado_gnuplot/gnuplot_window.hpp>
#include "includes/dynamics.h"

#ifndef PI
#define PI 3.141592
#endif

//#define DebugForce ;
//#define DebugActivation ;
//#define DebugLongueur ;
using namespace std;
USING_NAMESPACE_ACADO

/* ---------- Model ---------- */

biorbd::Model m("../../models/ModeleAv1Muscle.bioMod");

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
const double t_End= 1.0;
const int nPoints(30);

/* ---------- Functions ---------- */

#define  NX   nQ + nQdot        // number of differential states

#define  NOL  1                 // number of lagrange objective functions
void myLagrangeObjectiveFunction( double *x, double *g, void * ){
    g[0] = x[2]*x[2]+x[3];

}


#define  NOM   1                 // number of mayer objective functions
void myMayerObjectiveFunction( double *x, double *g, void * ){
    double obj = x[0]-PI/2;
    g[0] = obj*obj;
}

#define  NI   2                 // number of initial value constraints
void myInitialValueConstraint( double *x, double *g, void * ){
    g[0] = x[0]-0.01;
    g[1] = x[1];

}

#define  NE   1                 // number of end-point / terminal constraints
void myEndPointConstraint( double *x, double *g, void * ){
   // g[0]=x[0]-PI/2;                         // rotation de 90°
    g[0]=x[1];                              // vitesse nulle

}

int  main ()
{
    std::cout << "nb de marqueurs: " << nTags << std::endl<< std::endl;
    std::cout << "nb de muscles: " << nMus << std::endl<< std::endl;

    /* ---------- INITIALIZATION ---------- */
    Parameter               T;                              //  the  time  horizon T
    DifferentialState       x("",nQ+nQdot,1);               //  the  differential states
    Control                 u("", nMus, 1);                 //  the  control input  u
    IntermediateState       is(nQ + nQdot + nMus + 1);


    for (unsigned int i = 0; i < nQ; ++i)
        is(i) = x(i);
    for (unsigned int i = 0; i < nQdot; ++i)
        is(i+nQ) = x(i+nQ);
    for (unsigned int i = 0; i < nMus; ++i)
        is(i+nQ+nQdot) = u(i);
    is(3) = T;

    /* ----------- DEFINE OCP ------------- */
    OCP ocp( 0, 1 , nPoints);

    CFunction Mayer( NOM, myMayerObjectiveFunction);
    CFunction Lagrange( NOL, myLagrangeObjectiveFunction);
    ocp.minimizeMayerTerm( Mayer(is) );
    ocp.minimizeLagrangeTerm( Lagrange(is) );

    /* ------------ CONSTRAINTS ----------- */
    DifferentialEquation    f ;
    CFunction F( NX, forwardDynamicsFromMuscleActivation);
    CFunction I( NI, myInitialValueConstraint   );
    CFunction E( NE, myEndPointConstraint       );

    ocp.subjectTo( (f << dot(x)) == F(is)*T);                          //  differential  equation,
    ocp.subjectTo( AT_START, I(is) ==  0.0 );
    ocp.subjectTo( AT_END  , E(is) ==  0.0 );
    ocp.subjectTo(0.1 <= T <= 5);

    ocp.subjectTo(0.01 <= u <= 1);

    /* ---------- OPTIMIZATION  ------------ */
    OptimizationAlgorithm  algorithm( ocp ) ;       //  construct optimization  algorithm ,

    algorithm.initializeDifferentialStates("../Results/StatesAv1Muscle.txt");
    algorithm.initializeParameters("../Initialisation/T1Muscle.txt");
    algorithm.initializeControls("../Results/ControlsAv1Muscle.txt");


    GnuplotWindow window;                           //  visualize  the  results  in  a  Gnuplot  window
    window.addSubplot(  x ,  "STATES x" ) ;
    window.addSubplot( u ,  "CONTROL  u" ) ;
    window.addSubplot( T ,  "Time parameter T" ) ;
    algorithm << window;
    algorithm.solve();                              //  solve the problem .

    VariablesGrid param;
    algorithm.getParameters("../Results/ParametersAv1Muscle.txt");
    algorithm.getDifferentialStates("../Results/StatesAv1Muscle.txt");
    algorithm.getControls("../Results/ControlsAv1Muscle.txt");

    return 0;
}
