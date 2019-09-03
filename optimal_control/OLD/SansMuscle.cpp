#include <acado_optimal_control.hpp>
#include <bindings/acado_gnuplot/gnuplot_window.hpp>
#include "includes/dynamics.h"
using namespace std;
USING_NAMESPACE_ACADO

#ifndef PI
#define PI 3.141592
#endif

/* ---------- Model ---------- */

biorbd::Model m("../../models/Bras.bioMod");

unsigned int nQ(m.nbQ());               // states number
unsigned int nQdot(m.nbQdot());         // derived states number
unsigned int nTau(m.nbGeneralizedTorque());           // controls number
unsigned int nMarkers(m.nMarkers());          // markers number
unsigned int nMus(0);
unsigned int nPhases(1);
GeneralizedCoordinates Q(nQ), Qdot(nQdot), Qddot(nQdot);
GeneralizedTorque Tau(nTau);
std::vector<biorbd::muscles::StateDynamics> state(nMus); // controls

const double t_Start=0.0;
const double t_End= 10.0;
const int nPoints(30);


/* ---------- Functions ---------- */
#define  NX   nQ + nQdot        // number of differential states

void myLagrangeObjectiveFunction( double *x, double *g, void * ){
    g[0]=x[10]*x[10]+x[11]*x[11]-(x[12]*x[12])+x[13]*x[13]+x[14]*x[14];
//    g[1]=x[11]*x[11];
//    g[2]=-(x[12]*x[12]);
//    g[3]=x[13]*x[13];
//    g[4]=x[14]*x[14];
}

void myMayerObjectiveFunction( double *x, double *g, void *user_data ){
    g[0]=x[5]*x[5];

}

#define  NI   nQ + nQdot         // number of initial value constraints
void myInitialValueConstraint( double *x, double *g, void * ){
    for (unsigned int i =0; i<nQ + nQdot; ++i) {
        g[i] =  x[i]-0.1;
    }
}

#define  NE   1                 // number of end-point / terminal constraints
void myEndPointConstraint( double *x, double *g, void * ){
    g[0]=x[nQ-1]-PI/4;
}

int  main ()
{
    std::cout << "nb de muscles: " << nMus << std::endl;
    std::cout << "nb de degrés de liberté: " << nQ << std::endl;
    std::cout << "nb de torques: " << nTau << std::endl;

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
    //CFunction Mayer( 1, myMayerObjectiveFunction);
    CFunction Lagrange( 1, myLagrangeObjectiveFunction);
    OCP ocp( 0, 1 , nPoints);                        // time  horizon
    //ocp.minimizeMayerTerm( Mayer(is) );                    // Mayer term
    ocp.minimizeLagrangeTerm( Lagrange(is) );                    // Lagrange term


    /* ------------ CONSTRAINTS ----------- */
    DifferentialEquation    f;                             //  the  differential  equation
    CFunction F( NX, forwardDynamicsFromJointTorque);
    CFunction I( NI, myInitialValueConstraint   );
    CFunction E( NE, myEndPointConstraint       );

    ocp.subjectTo( (f << dot(x)) == F(is)*T );                          //  differential  equation,
    ocp.subjectTo( AT_START, I(is) ==  0.0 );
    ocp.subjectTo( AT_END  , E(is) ==  0.0 );
    ocp.subjectTo(-100 <= u <= 100);
    ocp.subjectTo(0.1 <= T <= 4);
    
    ocp.subjectTo(-PI/8 <= x(0) <= 0.1);
    ocp.subjectTo(-PI/2 <= x(1) <= 0.1);
    ocp.subjectTo(-PI/4 <= x(2) <= PI);
    ocp.subjectTo(-PI/2 <= x(3) <= PI/2);
    ocp.subjectTo(-0.1 <= x(4) <= PI);

    /* ---------- OPTIMIZATION  ------------ */
    OptimizationAlgorithm  algorithm( ocp ) ;       //  construct optimization  algorithm
    algorithm.set(MAX_NUM_ITERATIONS, 1000);
    //algorithm.set(KKT_TOLERANCE, 1e-10);
    //algorithm.set(INTEGRATOR_TOLERANCE, 1e-6);

    VariablesGrid u_init(nTau, Grid(t_Start, t_End, 2));
    for(unsigned int i=0; i<2; ++i){
        for(unsigned int j=0; j<nTau; ++j){
            u_init(i, j) = 0;
        }
    }
    algorithm.initializeControls(u_init);

    VariablesGrid x_init(nQ+nQdot, Grid(t_Start, t_End, 2));
    for(unsigned int i=0; i<nQ-1; ++i){
         x_init(0, i) = 0.1;
         x_init(1, i) = 0.1;
    }

    x_init(0, 4) = 0.2;
    x_init(1, 4) = 0.8;

    for(unsigned int i=nQ; i<nQdot; ++i){
         x_init(0, i) = 0.01;
         x_init(1, i) = 0.01;
    }
    algorithm.initializeDifferentialStates(x_init);

    GnuplotWindow window;                           //  visualize  the  results  in  a  Gnuplot  window
    window.addSubplot(  x ,  "DISTANCE x" ) ;
    window.addSubplot( u ,  "CONTROL  u" ) ;
    algorithm << window;
    algorithm.solve();                              //  solve the problem

    algorithm.getParameters("../Results/ParametersSansMuscle.txt");
    algorithm.getDifferentialStates("../Results/StatesSansMuscle.txt");
    algorithm.getControls("../Results/ControlsSansMuscle.txt");

    return 0;
}
