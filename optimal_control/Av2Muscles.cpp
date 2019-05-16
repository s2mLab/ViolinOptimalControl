#include <acado_optimal_control.hpp>
#include <bindings/acado_gnuplot/gnuplot_window.hpp>
#include "includes/dynamics.h"
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
const double t_End= 10.0;
const int nPoints(30);

/* ---------- Functions ---------- */

#define  NX   nQ + nQdot        // number of differential states

#define  NOL   nMus                 // number of lagrange objective functions
void myLagrangeObjectiveFunction( double *x, double *g, void *){
    for(unsigned int i=0; i<nMus; ++i)
        g[i] = x[i+nQ+nQdot];
}


#define  NOM   1                 // number of mayer objective functions
void myMayerObjectiveFunction( double *x, double *g, void *){
    g[0] = (x[0]-PI/2)*(x[0]-PI/2);
    //g[1] = (x[1]-PI/6)*(x[1]-PI/6);
}

#define  NI   nQ+nQdot                 // number of initial value constraints
void myInitialValueConstraint( double *x, double *g, void *){
    for(unsigned int i=0; i<nQ; ++i){
        g[i]=x[i]-0.01;
        g[i+nQ]=x[i+nQ];
    }
}

#define  NE   1                 // number of end-point / terminal constraints
void myEndPointConstraint( double *x, double *g, void *){
    g[0]=x[0]-PI/2;
}


int  main ()
{
    std::cout << "nb de muscles: " << nMus << std::endl<< std::endl;
    std::cout << "nb de degré de liberté: " << nQ << std::endl<< std::endl;


    /* ---------- INITIALIZATION ---------- */
    Parameter               T;                              //  the  time  horizon T
    DifferentialState       x("",nQ+nQdot,1);               //  the  differential states
    Control                 u("", nMus, 1);                 //  the  control input  u
    IntermediateState       is(nQ + nQdot + nMus);


    for (unsigned int i = 0; i < nQ; ++i)
        is(i) = x(i);
    for (unsigned int i = 0; i < nQdot; ++i)
        is(i+nQ) = x(i+nQ);
    for (unsigned int i = 0; i < nMus; ++i)
        is(i+nQ+nQdot) = u(i);

    /* ----------- DEFINE OCP ------------- */
    OCP ocp( t_Start, T , nPoints);

    CFunction Mayer( NOM, myMayerObjectiveFunction);
    CFunction Lagrange( NOL, myLagrangeObjectiveFunction);
    ocp.minimizeMayerTerm( T );
    ocp.minimizeLagrangeTerm( Lagrange(is) );

    /* ------------ CONSTRAINTS ----------- */
    DifferentialEquation    f(0.0, T) ;
    CFunction F( NX, forwardDynamicsFromMuscleActivation);
    CFunction I( NI, myInitialValueConstraint   );
    CFunction E( NE, myEndPointConstraint       );

    ocp.subjectTo( (f << dot(x)) == F(is) );                          //  differential  equation,
    ocp.subjectTo( AT_START, I(is) ==  0.0 );
    ocp.subjectTo( AT_END  , E(is) ==  0.0 );
    ocp.subjectTo(0.01 <= u <= 1);

    ocp.subjectTo(3.0 <= T <= 10.0);

    /* ---------- OPTIMIZATION  ------------ */
    OptimizationAlgorithm  algorithm( ocp ) ;       //  construct optimization  algorithm ,

    VariablesGrid u_init(nMus, Grid(t_Start, t_End, 2));
    VariablesGrid x_init(nQ+nQdot, Grid(t_Start, t_End, 2));

    for(unsigned int i=0; i<2; ++i){
        for(unsigned int j=0; j<nMus; ++j){
            u_init(i, j) = 0.1;
        }
    }

    for(unsigned int i=0; i<nQ; ++i){
         x_init(0, i) = 0.1;
         x_init(1, i) = 0.1;
    }
    for(unsigned int i=nQ; i<nQdot; ++i){
         x_init(0, i) = 0.0;
         x_init(1, i) = 0.0;
    }

    algorithm.initializeControls(u_init);
    algorithm.initializeDifferentialStates(x_init);

    GnuplotWindow window;                           //  visualize  the  results  in  a  Gnuplot  window
    window.addSubplot(  x ,  "STATES x" ) ;
    window.addSubplot( u ,  "CONTROL  u" ) ;
    algorithm << window;
    algorithm.solve();                              //  solve the problem

    algorithm.getDifferentialStates("../Results/StatesAv2Muscles.txt");
    algorithm.getControls("../Results/ControlsAv2Muscles.txt");

    return 0;
}
