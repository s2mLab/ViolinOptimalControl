#include <acado_optimal_control.hpp>
#include <bindings/acado_gnuplot/gnuplot_window.hpp>
USING_NAMESPACE_ACADO

const int nDim(3);

void LagrangeObjective( double *x, double *g, void *){
    g[0]=0;
    for (unsigned int i=0; i<nDim; ++i)
        g[0]+=(x[i]*x[i]);
}

void MayeurObjective( double *x, double *g, void *){
    g[0]=0;
    for (unsigned int i=0; i<nDim; ++i)
        g[0]+=(x[nDim+i]*x[nDim+i]);
}

void Position( double *x, double *g, void *){
    for (unsigned int i=0; i<nDim; ++i)
        g[i] = x[i];
}

void Velocity( double *x, double *g, void *){
    for (unsigned int i=0; i<nDim; ++i)
        g[i] = x[nDim+i];
}

void dynamic( double *x, double *rhs, void *){

    for (unsigned int i = 0; i<nDim; ++i){ // Assuming nQ == nQdot
        rhs[i] = x[nDim+i];
        rhs[i + nDim] = x[i+(2*nDim)];
    }
}

int  main ()
{
    /* ---------- INITIALIZATION ---------- */
    //Parameter T;
    DifferentialState       x1("", 2*nDim, 1);
    DifferentialState       x2("", 2*nDim, 1);
    DifferentialState       x3("", 2*nDim, 1);
    DifferentialState       x4("", 2*nDim, 1);

    Control                 u1("", nDim, 1);
    Control                 u2("", nDim, 1);
    Control                 u3("", nDim, 1);
    Control                 u4("", nDim, 1);

    DifferentialEquation    f;

    IntermediateState       is1("", 3*nDim, 1);
    IntermediateState       is2("", 3*nDim, 1);
    IntermediateState       is3("", 3*nDim, 1);
    IntermediateState       is4("", 3*nDim, 1);

    for (unsigned int i = 0; i < 2*nDim; ++i){
        is1(i) = x1(i);
        is2(i) = x2(i);
        is3(i) = x3(i);
        is4(i) = x4(i);
    }
    for (unsigned int i = 0; i < nDim; ++i){
        is1(i+2*nDim) = u1(i);
        is2(i+2*nDim) = u2(i);
        is3(i+2*nDim) = u3(i);
        is4(i+2*nDim) = u4(i);
    }

    /* ----------- DEFINE OCP ------------- */
    double t_end(40.0);
    OCP ocp(  0.0 , t_end , 30);

    CFunction Mayeur(1, MayeurObjective);
    CFunction Lagrange(1, LagrangeObjective);
    ocp.minimizeLagrangeTerm(Lagrange(u1) + Lagrange(u2) + Lagrange(u3) + Lagrange(u4));
    ocp.minimizeMayerTerm(Mayeur(x1) + Mayeur(x2) + Mayeur(x3) + Mayeur(x4));

    CFunction F(2*nDim, dynamic);
    (f << dot( x1 )) == F(is1);
    (f << dot( x2 )) == F(is2);
    (f << dot( x3 )) == F(is3);
    (f << dot( x4 )) == F(is4);

    /* ------------ CONSTRAINTS ----------- */

    ocp.subjectTo( f ) ;

    CFunction Pos( nDim, Position);
    CFunction Vel( nDim, Velocity);
    ocp.subjectTo( AT_START,  x1 ==  0.0 ) ;
    ocp.subjectTo( AT_END   ,  Pos(x1) == 100.0) ;
    ocp.subjectTo(0.0, x2, -x1, 0.0);
    ocp.subjectTo( AT_END   ,  Pos(x2) == 0.0) ;
    ocp.subjectTo(0.0, x3, -x2, 0.0);
    ocp.subjectTo( AT_END   ,  Pos(x3) == 100.0) ;
    ocp.subjectTo(0.0, x4, -x3, 0.0);
     ocp.subjectTo( AT_END   ,  Pos(x4) == 0.0) ;

    ocp.subjectTo( -100.0 <= Pos(x1) <= 200.0);
    ocp.subjectTo( -100.0 <= Pos(x2) <= 200.0);
    ocp.subjectTo( -100.0 <= Pos(x3) <= 200.0);
    ocp.subjectTo( -100.0 <= Pos(x4) <= 200.0);

    ocp.subjectTo( -10.0 <= Vel(x1) <= 10.0);
    ocp.subjectTo( -10.0 <= Vel(x2) <= 10.0);
    ocp.subjectTo( -10.0 <= Vel(x3) <= 10.0);
    ocp.subjectTo( -10.0 <= Vel(x4) <= 10.0);


    ocp.subjectTo( -5 <= u1 <= 5);
    ocp.subjectTo( -5 <= u2 <= 5);
    ocp.subjectTo( -5 <= u3 <= 5);
    ocp.subjectTo( -5 <= u4 <= 5);

    //ocp.subjectTo(0.0 <= T <= 60);

    /* ---------- VISUALIZATION ------------ */

    GnuplotWindow window;                           //  visualize  the  results  in  a  Gnuplot  window
    window.addSubplot(  x1 ,  "DISTANCE x1" ) ;
    window.addSubplot(  x2 ,  "DISTANCE x2" ) ;
    window.addSubplot(  x3 ,  "DISTANCE x3" ) ;
    window.addSubplot(  x4 ,  "DISTANCE x4" ) ;
    window.addSubplot( u1 ,  "CONTROL  u1" ) ;
    window.addSubplot( u2 ,  "CONTROL  u2" ) ;
    window.addSubplot( u3 ,  "CONTROL  u3" ) ;
    window.addSubplot( u4 ,  "CONTROL  u4" ) ;

    /* ---------- OPTIMIZATION  ------------ */

    OptimizationAlgorithm  algorithm( ocp ) ;       //  construct optimization  algorithm ,
    algorithm.set(MAX_NUM_ITERATIONS, 1000);

    algorithm << window;
    algorithm.solve();

    algorithm.getDifferentialStates("../Results/StatesEocar.txt");
    algorithm.getControls("../Results/ControlsEocar.txt");

    return  0;
}
