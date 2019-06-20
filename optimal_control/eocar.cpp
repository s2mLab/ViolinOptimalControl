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

s2mMusculoSkeletalModel m("../../models/eocar.bioMod");
unsigned int nQ(m.nbQ());               // states number
unsigned int nQdot(m.nbQdot());         // derived states number
unsigned int nTau(m.nbTau());           // controls number
unsigned int nTags(m.nTags());          // markers number
unsigned int nMus(m.nbMuscleTotal());   // muscles number

const double t_Start = 0.0;
const double t_End = 5.0;
const int nPoints(30);

void LagrangeObjective( double *x, double *g, void *){
    g[0]=0;
    for (unsigned int i=0; i<nQ; ++i)
        g[0]+=(x[i]*x[i]);
}

void MayeurObjective( double *x, double *g, void *){
    g[0]=0;
    for (unsigned int i=0; i<nQ; ++i)
        g[0]+=(x[nQ+i]*x[nQ+i]);
}

void Position( double *x, double *g, void *){
    for (unsigned int i=0; i<nQ; ++i)
        g[i] = x[i];
}

void Velocity( double *x, double *g, void *){
    for (unsigned int i=0; i<nQ; ++i)
        g[i] = x[nQ+i];
}

void dynamic( double *x, double *rhs, void *){

    for (unsigned int i = 0; i<nQ; ++i){ // Assuming nQ == nQdot
        rhs[i] = x[i+nQ];
        rhs[i + nQdot] = x[i+2*nQ];
    }
}

int  main ()
{
    /* ---------- INITIALIZATION ---------- */
    DifferentialState       x1("", 2*nQ, 1);
    DifferentialState       x2("", 2*nQ, 1);
    Control                 u1("", nQ, 1);
    Control                 u2("", nQ, 1);
    DifferentialEquation    f;
    IntermediateState       is1("", 3*nQ, 1);
    IntermediateState       is2("", 3*nQ, 1);


    for (unsigned int i = 0; i < 2*nQ; ++i){
        is1(i) = x1(i);
        is2(i) = x2(i);
    }
    for (unsigned int i = 0; i < nQ; ++i){
        is1(i+2*nQ) = u1(i);
        is2(i+2*nQ) = u2(i);
    }

    /* ----------- DEFINE OCP ------------- */
    OCP ocp(  0.0 , t_End , 30);

    CFunction Mayeur(1, MayeurObjective);
    CFunction Lagrange(1, LagrangeObjective);
    ocp.minimizeLagrangeTerm(Lagrange(u1) + Lagrange(u2));
    ocp.minimizeMayerTerm(Mayeur(x1) + Mayeur(x2));

    CFunction F(2*nQ, forwardDynamicsFromJointTorque);
    (f << dot( x1 )) == F(is1);
    (f << dot( x2 )) == F(is2);

    /* ------------ CONSTRAINTS ----------- */

    ocp.subjectTo( f ) ;

    CFunction Pos( nQ, Position);
    CFunction Vel( nQ, Velocity);
    ocp.subjectTo( AT_START,  x1 ==  0.0 ) ;
    ocp.subjectTo( AT_END   ,  Pos(x1) == 3.1415) ;
    ocp.subjectTo(0.0, x2, -x1, 0.0);
    ocp.subjectTo( AT_END   ,  Pos(x2) == 0.0) ;

    ocp.subjectTo( -1 <= Pos(x1) <= 4);
    ocp.subjectTo( -1 <= Pos(x2) <= 4);

    ocp.subjectTo( -5.0 <= Vel(x1) <= 5.0);
    ocp.subjectTo( -5.0 <= Vel(x2) <= 5.0);

    ocp.subjectTo( -2 <= u1 <= 2);
    ocp.subjectTo( -2 <= u2 <= 2);

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
