#include <acado_optimal_control.hpp>
#include <bindings/acado_gnuplot/gnuplot_window.hpp>
#include <vector>

using namespace std;
USING_NAMESPACE_ACADO

/* ---------- Model ---------- */
const double t_Start = 0.0;
const double t_End = 5.;
const int nPoints(31);

void dynamics(double *x, double *rhs, void *user_data){
    rhs[0] = 2*x[1];
    rhs[1] = 3*x[2];
}

void dyn_deriv(int n, double *x, double *seed, double *f, double *df, void *user_data){
    dynamics(seed, df, user_data);
}

int  main ()
{
    /* ---------- INITIALIZATION ---------- */
    DifferentialState       x("", 2, 1);
    Control                 u("", 1, 1);
    IntermediateState       is("", 3, 1);
    DifferentialEquation    f;
    is(0) = x(0);
    is(1) = x(1);
    is(2) = u(0);

    /* ----------- DEFINE OCP ------------- */
    OCP ocp(  0.0 , t_End , 30);
    CFunction dyn(2, dynamics, dyn_deriv, dyn_deriv);

    ocp.minimizeLagrangeTerm( u*u );
    (f << dot( x )) == dyn(is);
    ocp.subjectTo( f ) ;

    /* ------------ CONSTRAINTS ----------- */
    ocp.subjectTo( AT_START, x(0) ==  0 );
    ocp.subjectTo( AT_START, x(1) ==  0 );

    ocp.subjectTo( AT_END, x(0) ==  100 );
    ocp.subjectTo( AT_END, x(1) ==  0 );

    ocp.subjectTo( -500 <= x(0) <= 500);
    ocp.subjectTo( -500 <= x(1) <= 500);

    ocp.subjectTo( -100 <= u(0) <= 100);

    /* ---------- VISUALIZATION ------------ */
    GnuplotWindow window;                           //  visualize  the  results  in  a  Gnuplot  window
    window.addSubplot(  x(0) ,  "Position" ) ;
    window.addSubplot(  x(1) ,  "Velocity" ) ;
    window.addSubplot( u(0) ,  "Acceleration" ) ;

    /* ---------- OPTIMIZATION  ------------ */
    OptimizationAlgorithm  algorithm( ocp ) ;       //  construct optimization  algorithm ,
    algorithm.set(MAX_NUM_ITERATIONS, 1000);

    algorithm << window;
    algorithm.solve();

    return  0;
}
