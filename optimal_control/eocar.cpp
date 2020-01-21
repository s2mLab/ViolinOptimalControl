#include <acado_optimal_control.hpp>
#include <bindings/acado_gnuplot/gnuplot_window.hpp>
#include <vector>
#include <memory>

USING_NAMESPACE_ACADO

// ---------- Model ---------- //

const double t_Start = 0.0;
const double t_End = 2.0;
const int nPoints(31);

const int nx(2);
const int nu(1);

void finalPosition(double *x, double *g, void *){
    g[0] = 10 - x[0];
    g[1] = x[1];
}

void dynamics(double *is, double *g, void *){
    g[0] = is[2];
    g[1] = is[0];
}

int  main(){
    CFunction cFinalPosition(2, finalPosition);
    CFunction cDynamics(2, dynamics);

    clock_t start = clock();

    // ----------- DEFINE OCP ------------- //
    OCP ocp(t_Start, t_End, nPoints);

    // ------------ CONSTRAINTS ----------- //
    Control u("", nu, 1);
    DifferentialState x("",nx,1);
    IntermediateState is(nu + nx);

    for (unsigned int i = 0; i < nu; ++i)
        is(i) = u(i);
    for (unsigned int i = 0; i < nx; ++i)
        is(i+nu) = x(i);
    DifferentialEquation f;
    f << dot(x);
    ocp.subjectTo( f == cDynamics(is) );

    ocp.subjectTo( AT_START, x ==  0 );
//    ocp.subjectTo( AT_END, cFinalPosition(x) == 0.0);
    ocp.subjectTo( AT_END, x(0) == 10.0);
    ocp.subjectTo( AT_END, x(1) == 0.0);

    for (unsigned int i=0; i<nu; ++i)
        ocp.subjectTo(-100 <= u(i) <= 100);
    for (unsigned int i=0; i<nx; ++i)
        ocp.subjectTo(-100 <= x(i) <= 100);

    // ------------ OBJECTIVE ----------- //
    Expression sumLagrange(u*u);
    ocp.minimizeLagrangeTerm(sumLagrange);

    // ---------- VISUALIZATION ------------ //
    GnuplotWindow window;
    window.addSubplot( x(0), "Position" );
    window.addSubplot( x(1), "Vitesse" );
    window.addSubplot( u(0),  "Acceleration" );


    // ---------- OPTIMIZATION  ------------ //
    OptimizationAlgorithm  algorithm( ocp ) ;
    algorithm << window;
    algorithm.solve();

    // ---------- SHOW SOLUTION  ------------ //
    VariablesGrid finalU, finalX;
    algorithm.getControls(finalU);
    algorithm.getDifferentialStates(finalX);
//    finalU.print();
//    finalX.print();


    // ---------- FINALIZE  ------------ //
    clock_t end=clock();
    double time_exec(double(end - start)/CLOCKS_PER_SEC);
    std::cout<<"Execution time: "<<time_exec<<std::endl;
    return  0;
}
