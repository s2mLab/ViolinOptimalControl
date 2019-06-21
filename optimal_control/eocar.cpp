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

void Position( double *x, double *g, void *){
    for (unsigned int i=0; i<nQ; ++i)
        g[i] = x[i];
}

void Velocity( double *x, double *g, void *){
    for (unsigned int i=0; i<nQ; ++i)
        g[i] = x[nQ+i];
}

int  main ()
{
    /* ---------- INITIALIZATION ---------- */
    DifferentialState       x1("", nQ+nQdot, 1);
    DifferentialState       x2("", nQ+nQdot, 1);
    Control                 u1("", nTau, 1);
    Control                 u2("", nTau, 1);
    DifferentialEquation    f;
    IntermediateState       is1("", nQ+nQdot+nTau, 1);
    IntermediateState       is2("", nQ+nQdot+nTau, 1);


    for (unsigned int i = 0; i < nQ; ++i){
        is1(i) = x1(i);
        is2(i) = x2(i);
    }
    for (unsigned int i = 0; i < nQdot; ++i){
        is1(i+nQ) = x1(i+nQ);
        is2(i+nQ) = x2(i+nQ);
    }
    for (unsigned int i = 0; i < nTau; ++i){
        is1(i+nQ+nQdot) = u1(i);
        is2(i+nQ+nQdot) = u2(i);
    }

    /* ----------- DEFINE OCP ------------- */
    OCP ocp(  0.0 , t_End , 30);

    CFunction Mayer(1, MayerVelocity);
    CFunction Lagrange(1, LagrangeResidualTorques);
    ocp.minimizeLagrangeTerm(Lagrange(is1) + Lagrange(is2));
    ocp.minimizeMayerTerm(Mayer(is1) + Mayer(is2));

    CFunction F(2*nQ, forwardDynamicsFromJointTorque);
    (f << dot( x1 )) == F(is1);
    (f << dot( x2 )) == F(is2);

    /* ------------ CONSTRAINTS ----------- */

    ocp.subjectTo( f ) ;

    CFunction Pos( nQ, Position);
    CFunction Vel( nQ, Velocity);
    CFunction Frog( 4, ViolonUp);
    CFunction Tip( 4,  ViolonDown);
    CFunction Velocity(nQdot, VelocityZero);

    ocp.subjectTo( AT_START, Frog(x1) ==  0.0 );
    ocp.subjectTo( AT_END  , Tip(x1) ==  0.0 );
    ocp.subjectTo( 0.0, x2, -x1, 0.0 );
    //ocp.subjectTo( AT_START, Tip(x2) ==  0.0 );
    ocp.subjectTo(AT_END, Tip(x2) == 0.0);

    ocp.subjectTo(AT_START, Velocity(x1) == 0.0);

    ocp.subjectTo( -4 <= Pos(x1) <= 4);
    ocp.subjectTo( -4 <= Pos(x2) <= 4);

    ocp.subjectTo( -5.0 <= Vel(x1) <= 5.0);
    ocp.subjectTo( -5.0 <= Vel(x2) <= 5.0);

    ocp.subjectTo( -100 <= u1 <= 100);
    ocp.subjectTo( -100 <= u2 <= 100);

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
