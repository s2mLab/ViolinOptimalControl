#include <acado_optimal_control.hpp>
#include <bindings/acado_gnuplot/gnuplot_window.hpp>
#include <vector>
#include <memory>

USING_NAMESPACE_ACADO

// ---------- Model ---------- //

const double t_Start = 0.0;
const double t_End = 2.0;
const int nPoints(30);

void finalPosition(double *x, double *g, void *){
    g[0] = 10 - x[0];
    g[1] = 10 - x[1];
    g[2] = x[2];
    g[3] = x[3];
}

void dynamics(double *is, double *g, void *){
    g[0] = is[2];
    g[1] = is[3];
    g[2] = is[4]-9.8;
    g[3] = is[5]-9.8;
}

int  main(){
    int nx(4);
    int nu(2);
    CFunction cFinalPosition(nx, finalPosition);
    CFunction cDynamics(nx, dynamics);

    clock_t start = clock();

    // ----------- DEFINE OCP ------------- //
    OCP ocp(t_Start, t_End, nPoints);

    // ------------ CONSTRAINTS ----------- //
    DifferentialState x("", nx, 1);
    Control u("", nu, 1);
    IntermediateState is(nx + nu);

    for (unsigned int i = 0; i < nx; ++i)
        is(i) = x(i);
    for (unsigned int i = 0; i < nu; ++i)
        is(i+nx) = u(i);
    DifferentialEquation f;
    (f << dot(x)) == cDynamics(is);
    ocp.subjectTo( f );

//    ocp.subjectTo( AT_START, x ==  0 );
//    ocp.subjectTo( AT_END, cFinalPosition(is) == 0);

//    for (unsigned int i=0; i<nx; ++i)
//        ocp.subjectTo(-100 <= x(i) <= 100);
//    for (unsigned int i=0; i<nu; ++i)
//        ocp.subjectTo(-100 <= u(i) <= 100);

    // ------------ OBJECTIVE ----------- //
    Expression sumLagrange(u*u);
    ocp.minimizeLagrangeTerm(sumLagrange);

    // ---------- VISUALIZATION ------------ //
    GnuplotWindow window;
    for (int i=0; i<nx/2;  ++i){
        window.addSubplot( x(i), "Position" );
    }
    for (int i=nx/2; i<nx;  ++i){
        window.addSubplot( x(i), "Vitesse" );
    }
    for (int i=0; i<nu;  ++i){
        window.addSubplot( u(i),  "Acceleration" );
    }


    // ---------- OPTIMIZATION  ------------ //
    OptimizationAlgorithm  algorithm( ocp ) ;
    algorithm.set(INTEGRATOR_TYPE, INT_RK45);
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
