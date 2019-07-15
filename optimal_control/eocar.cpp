#include <acado_optimal_control.hpp>
#include <bindings/acado_gnuplot/gnuplot_window.hpp>
#include "s2mMusculoSkeletalModel.h"
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
unsigned int nTau(m.nbTau());           // controls number
unsigned int nTags(m.nTags());          // markers number
unsigned int nMus(m.nbMuscleTotal());   // muscles number

const double t_Start = 0.0;
const double t_End = 1.0;
const int nPoints(31);

void position( double *x, double *g, void *){
    for (unsigned int i=0; i<nQ; ++i)
        g[i] = x[i];
}

void velocity( double *x, double *g, void *){
    for (unsigned int i=0; i<nQ; ++i)
        g[i] = x[nQ+i];
}

int  main ()
{
    /* ---------- INITIALIZATION ---------- */
    DifferentialState       x1("", nQ+nQdot, 1);
    DifferentialState       x2("", nQ+nQdot, 1);
    Control                 u1("", nMus+nTau, 1);
    Control                 u2("", nMus+nTau, 1);
    DifferentialEquation    f;
    IntermediateState       is1("", nQ+nQdot+nMus+nTau, 1);
    IntermediateState       is2("", nQ+nQdot+nMus+nTau, 1);


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
        is1(i+nQ+nQdot+nMus) = u1(i+nMus);
        is2(i+nQ+nQdot+nMus) = u2(i+nMus);
    }

    /* ----------- DEFINE OCP ------------- */
    OCP ocp(  0.0 , t_End , 30);

    CFunction mayer(1, mayerVelocity);
    CFunction lagrange(1, lagrangeResidualTorques);
    ocp.minimizeLagrangeTerm(lagrange(is1) + lagrange(is2));
    ocp.minimizeMayerTerm(mayer(is1) + mayer(is2));

    CFunction F(2*nQ, forwardDynamicsFromJointTorque);
    (f << dot( x1 )) == F(is1);
    (f << dot( x2 )) == F(is2);

    /* ------------ CONSTRAINTS ----------- */

    ocp.subjectTo( f ) ;

    CFunction pos( nQ, position);
    CFunction vel( nQ, velocity);
    CFunction velocity(nQdot, velocityZero);

    ocp.subjectTo( AT_START, x1(1) ==  -1.13 );
    ocp.subjectTo( AT_START, x1(2) ==  0.61 );
    ocp.subjectTo( AT_START, x1(3) ==  -0.35 );
    ocp.subjectTo( AT_START, x1(4) ==  1.55 );

    ocp.subjectTo( AT_END, x1(1) ==  -0.7 );
    ocp.subjectTo( AT_END, x1(2) ==  0.17 );
    ocp.subjectTo( AT_END, x1(3) ==  0.0 );
    ocp.subjectTo( AT_END, x1(4) ==  0.61 );

    ocp.subjectTo( 0.0, x2, -x1, 0.0 );

    ocp.subjectTo( AT_END, x2(1) ==  -0.7 );
    ocp.subjectTo( AT_END, x2(2) ==  0.17 );
    ocp.subjectTo( AT_END, x2(3) ==  0.0 );
    ocp.subjectTo( AT_END, x2(4) ==  0.61 );

    ocp.subjectTo(AT_START, velocity(x1) == 0.0);

    ocp.subjectTo( -4 <= pos(x1) <= 4);
    ocp.subjectTo( -4 <= pos(x2) <= 4);

    ocp.subjectTo( -5.0 <= vel(x1) <= 5.0);
    ocp.subjectTo( -5.0 <= vel(x2) <= 5.0);

    for (unsigned int i=0; i<nMus; ++i){
         ocp.subjectTo(0.01 <= u1(i) <= 1);
         ocp.subjectTo(0.01 <= u2(i) <= 1);
    }

    for (unsigned int i=nMus; i<nMus+nTau; ++i){
         ocp.subjectTo(-100 <= u1(i) <= 100);
         ocp.subjectTo(-100 <= u2(i) <= 100);
    }


    /* ---------- VISUALIZATION ------------ */

    GnuplotWindow window;                           //  visualize  the  results  in  a  Gnuplot  window
    window.addSubplot(  x1 ,  "DISTANCE x1" ) ;
    window.addSubplot(  x2 ,  "DISTANCE x2" ) ;
    window.addSubplot( u1 ,  "CONTROL  u1" ) ;
    window.addSubplot( u2 ,  "CONTROL  u2" ) ;

    /* ---------- OPTIMIZATION  ------------ */

    OptimizationAlgorithm  algorithm( ocp ) ;       //  construct optimization  algorithm ,
    algorithm.set(MAX_NUM_ITERATIONS, 1000);

    algorithm << window;
    algorithm.solve();

    algorithm.getDifferentialStates("../Results/StatesEocar.txt");
    algorithm.getControls("../Results/ControlsEocar.txt");

    return  0;
}
