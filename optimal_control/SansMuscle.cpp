#include <acado_optimal_control.hpp>
#include <bindings/acado_gnuplot/gnuplot_window.hpp>
#include "includes/dynamics.h"
using namespace std;
USING_NAMESPACE_ACADO

/* ---------- Model ---------- */

s2mMusculoSkeletalModel m("../Modeles/ModeleSansMuscle.bioMod");

unsigned int nQ(m.nbQ());               // states number
unsigned int nQdot(m.nbQdot());         // derived states number
unsigned int nTau(m.nbTau());           // controls number
unsigned int nTags(m.nTags());          // markers number
unsigned int nMus(0);

const double t_Start=0.0;
const double t_End= 10.0;
const int nPoints(30);


/* ---------- Functions ---------- */
#define  NX   nQ + nQdot        // number of differential states

#define  NOL   1                 // number of lagrange objective functions
void myLagrangeObjectiveFunction( double *, double *g, void * ){
    g[0] = 0;
}

#define  NOM   1                 // number of mayer objective functions
void myMayerObjectiveFunction( double *x, double *g, void * ){
    double obj = x[0]-PI/2;
    g[0] = obj*obj;
}

#define  NI   nQ + nQdot         // number of initial value constraints
void myInitialValueConstraint( double *x, double *g, void * ){
    for (unsigned int i =0; i<nQ + nQdot; ++i) {
        g[i] =  x[i];
    }
}

#define  NE   1                 // number of end-point / terminal constraints
void myEndPointConstraint( double *x, double *g, void * ){
    // g[0]=x[0]-PI/2;                         // rotation de 90°
    g[0]=x[1];                              // vitesse nulle
}


int  main ()
{
    /* ---------- INITIALIZATION ---------- */
    Parameter               T;                              //  the  time  horizon T
    DifferentialState       x("",nQ+nQdot,1);               //  the  differential states
    Control                 u("", nTau, 1);          //  the  control input  u
    IntermediateState       is(nQ + nQdot + nTau);

    for (unsigned int i = 0; i < nQ; ++i){ // assuming nQ == nQdot
        is(i) = x(i);                   //*scalingQ(i);
        is(i+nQ) = x(i+nQ);             //*scalingQdot(i);
    }
    for (unsigned int i = 0; i < nTau; ++i)
        is(i+nQ+nQdot) = u(i);          //*scalingQdot(i);

    /* ----------- DEFINE OCP ------------- */
    CFunction Mayer( NOM, myMayerObjectiveFunction);
    CFunction Lagrange( NOL, myLagrangeObjectiveFunction);
    OCP ocp( t_Start, T , nPoints);                        // time  horizon
    ocp.minimizeMayerTerm( Mayer(is) );                    // Mayer term
    ocp.minimizeLagrangeTerm( Lagrange(is) );                    // Lagrange term


    /* ------------ CONSTRAINTS ----------- */
    DifferentialEquation    f;                             //  the  differential  equation
    CFunction F( NX, forwardDynamicsFromJointTorque);
    CFunction I( NI, myInitialValueConstraint   );
    CFunction E( NE, myEndPointConstraint       );

    ocp.subjectTo( (f << dot(x)) == F(is) );                          //  differential  equation,
    ocp.subjectTo( AT_START, I(is) ==  0.0 );
    ocp.subjectTo( AT_END  , E(is) ==  0.0 );
    ocp.subjectTo(-100 <= u <= 100);



    /* ---------- OPTIMIZATION  ------------ */
    OptimizationAlgorithm  algorithm( ocp ) ;       //  construct optimization  algorithm ,

    VariablesGrid u_init(1, Grid(t_Start, t_End, 2));
    u_init(0, 0) = 0.1;
    u_init(0, 1) = 0.1;
    algorithm.initializeControls(u_init);

    VariablesGrid x_init(2, Grid(t_Start, t_End, 2));
    x_init(0, 0) = 0.1;
    x_init(0, 1) = 0.1;
    x_init(1, 0) = 0.1;
    x_init(1, 1) = 0.1;
    algorithm.initializeDifferentialStates(x_init);

    GnuplotWindow window;                           //  visualize  the  results  in  a  Gnuplot  window
    window.addSubplot(  x ,  "DISTANCE x" ) ;
    window.addSubplot( u ,  "CONTROL  u" ) ;
    algorithm << window;
    algorithm.solve();                              //  and solve the problem .

    algorithm.getDifferentialStates("../Results/StatesSansMuscle.txt");
    algorithm.getControls("../Results/ControlsSansMuscle.txt");

    return 0;
}


