#include <acado_optimal_control.hpp>
#include <bindings/acado_gnuplot/gnuplot_window.hpp>
#include "includes/dynamics.h"
#include "includes/objectives.h"
#include "includes/constraints.h"
#include "includes/utils.h"
#include <time.h>

using namespace std;
USING_NAMESPACE_ACADO

s2mMusculoSkeletalModel m("../../models/BrasViolon.bioMod");
unsigned int nQ(m.nbQ());               // states number
unsigned int nQdot(m.nbQdot());         // derived states number
unsigned int nTau(m.nbTau());           // controls number
unsigned int nTags(m.nTags());          // markers number
unsigned int nMus(m.nbMuscleTotal());   // muscles number

const double t_Start = 0.0;
const double t_End = 0.5;
const int nPoints(31);

int  main ()
{
    std::string resultsPath("../Results/");

    clock_t start,end;
    double time_exec;
    start=clock();

    std::cout << "nb de muscles: " << nMus << std::endl;
    std::cout << "nb de torques: " << nTau << std::endl;
    std::cout << "nb de marqueurs: " << nTags << std::endl;

    /* ---------- INITIALIZATION ---------- */
    DifferentialState       x1("",nQ+nQdot,1);               //  the  differential states
    DifferentialState       x2("",nQ+nQdot,1);
    Control                 u1("", nMus+nTau, 1);                 //  the  control input  u
    Control                 u2("", nMus+nTau, 1);
    IntermediateState       is1(nQ + nQdot + nMus + nTau); // + 1);
    IntermediateState       is2(nQ + nQdot + nMus + nTau); // + 1);

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
    OCP ocp(t_Start, t_End, nPoints);

    CFunction Lagrange(1, LagrangeResidualTorques);
    ocp.minimizeLagrangeTerm( Lagrange(is1) + Lagrange(is2));

    /* ------------ CONSTRAINTS ----------- */
    CFunction F( nQ+nQdot, forwardDynamicsFromMuscleActivationAndTorque);

    DifferentialEquation    f ;
    (f << dot(x1)) == F(is1);
    (f << dot(x2)) == F(is2);
    ocp.subjectTo(f);

    //m.AddContactConstraint()
    int tagArchetPoucette = 16;
    int tagArchetCOM = 17;
    int tagArchetTete = 18;
    int tagViolon = 34;
    CFunction MarkerArchetPoucette(3, MarkerPosition);
    MarkerArchetPoucette.setUserData((void*) &tagArchetPoucette);
    CFunction MarkerArchetCOM(3, MarkerPosition);
    MarkerArchetCOM.setUserData((void*) &tagArchetCOM);
    CFunction MarkerArchetTete(3, MarkerPosition);
    MarkerArchetTete.setUserData((void*) &tagArchetTete);
    CFunction MarkerViolon(3, MarkerPosition);
    MarkerViolon.setUserData((void*) &tagViolon);

    ocp.subjectTo( AT_START, MarkerArchetPoucette(x1) - MarkerViolon(x1) == 0.0 );
    ocp.subjectTo( AT_END, MarkerArchetTete(x1) - MarkerViolon(x1) == 0.0 );
    ocp.subjectTo( 0.0, x2, -x1, 0.0 );
    ocp.subjectTo( 0.0, x1, -x2, 0.0 );

    for (unsigned int i=0; i<nMus; ++i){
         ocp.subjectTo(0.01 <= u1(i) <= 1);
         ocp.subjectTo(0.01 <= u2(i) <= 1);
    }

    for (unsigned int i=nMus; i<nMus+nTau; ++i){
         ocp.subjectTo(-100 <= u1(i) <= 100);
         ocp.subjectTo(-100 <= u2(i) <= 100);
    }

    ocp.subjectTo(-PI/8 <= x1(0) <= 0.1);
    ocp.subjectTo(-PI/2 <= x1(1) <= 0.1);
    ocp.subjectTo(-PI/4 <= x1(2) <= PI);
    ocp.subjectTo(-PI/2 <= x1(3) <= PI/2);
    ocp.subjectTo(-0.1 <= x1(4) <= PI);

    ocp.subjectTo(-PI/8 <= x2(0) <= 0.1);
    ocp.subjectTo(-PI/2 <= x2(1) <= 0.1);
    ocp.subjectTo(-PI/4 <= x2(2) <= PI);
    ocp.subjectTo(-PI/2 <= x2(3) <= PI/2);
    ocp.subjectTo(-0.1 <= x2(4) <= PI);

    /* ---------- OPTIMIZATION  ------------ */
    OptimizationAlgorithm  algorithm( ocp ) ;       //  construct optimization  algorithm ,
    algorithm.set(MAX_NUM_ITERATIONS, 1000);
    algorithm.set(INTEGRATOR_TYPE, INT_RK45);
    algorithm.set(HESSIAN_APPROXIMATION, FULL_BFGS_UPDATE);
    algorithm.set(KKT_TOLERANCE, 1e-4);

    /* ---------- INITIAL SOLUTION ---------- */
    VariablesGrid u_init(2*(nTau + nMus), Grid(t_Start, t_End, 2));
    for(unsigned int i=0; i<2; ++i){
        for(unsigned int j=0; j<nMus; ++j){
            u_init(i, j) = 0.1;
        }
        for(unsigned int j=nMus; j<nMus+nTau; ++j){
            u_init(i, j) = 0.01;
        }
        for(unsigned int j=nMus+nTau; j<(2*nMus)+nTau; ++j){
            u_init(i, j) = 0.1;
        }
        for(unsigned int j=(2*nMus)+nTau; j<2*(nMus+nTau); ++j){
            u_init(i, j) = 0.01;
        }
    }
    algorithm.initializeControls(u_init);

    VariablesGrid x_init(2*(nQ+nQdot), Grid(t_Start, t_End, 2));

    x_init(0, 0) = 0.09973;
    x_init(0, 1) = 0.09733;
    x_init(0, 2) = 1.05710;
    x_init(0, 3) = 1.56950;
    x_init(0, 4) = 1.07125;
    x_init(0, 5) = 0.95871;
    x_init(0, 6) = -1.7687;

    x_init(1, 0) = -0.39107;
    x_init(1, 1) = -0.495383;
    x_init(1, 2) = -0.089030;
    x_init(1, 3) = 0.1485315;
    x_init(1, 4) = 0.8569764;
    x_init(1, 5) = 1.9126840;
    x_init(1, 6) = -0.490220;

    x_init(0, nQ+nQdot) = -0.39107;
    x_init(0, 1+nQ+nQdot) = -0.495383;
    x_init(0, 2+nQ+nQdot) = -0.089030;
    x_init(0, 3+nQ+nQdot) = 0.1485315;
    x_init(0, 4+nQ+nQdot) = 0.8569764;
    x_init(0, 5+nQ+nQdot) = 1.9126840;
    x_init(0, 6+nQ+nQdot) = -0.490220;

    x_init(1, nQ+nQdot) = 0.09973;
    x_init(1, 1+nQ+nQdot) = 0.09733;
    x_init(1, 2+nQ+nQdot) = 1.05710;
    x_init(1, 3+nQ+nQdot) = 1.56950;
    x_init(1, 4+nQ+nQdot) = 1.07125;
    x_init(1, 5+nQ+nQdot) = 0.95871;
    x_init(1, 6+nQ+nQdot) = -1.7687;

    for(unsigned int i=nQ; i<nQ+nQdot; ++i){
         x_init(0, i) = 0.01;
         x_init(1, i) = 0.01;
         x_init(0, i+nQ+nQdot) = 0.01;
         x_init(1, i+nQ+nQdot) = 0.01;

    }
    algorithm.initializeDifferentialStates(x_init);

    /* ---------- SOLVING THE PROBLEM ---------- */
    algorithm.solve();


    /* ---------- STORING THE RESULTS ---------- */
    createTreePath(resultsPath);
    algorithm.getDifferentialStates((resultsPath + "StatesAv2Phases.txt").c_str());
    algorithm.getControls((resultsPath + "ControlsAv2Phases.txt").c_str());

    end=clock();
    time_exec = double(end - start)/CLOCKS_PER_SEC;
    std::cout<<"Execution time: "<<time_exec<<std::endl;

    return 0;
}
