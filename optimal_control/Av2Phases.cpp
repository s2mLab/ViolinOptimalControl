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
const int nPoints(45);

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
    CFunction F( nQ+nQdot, forwardDynamicsFromMuscleActivationAndTorqueContact);

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

//    ocp.subjectTo( AT_START, MarkerArchetPoucette(x1) - MarkerViolon(x1) == 0.0 );
    ocp.subjectTo( AT_START, MarkerArchetCOM(x1) - MarkerViolon(x1) == 0.0 );
    ocp.subjectTo( AT_END, MarkerArchetCOM(x1) - MarkerViolon(x1) == 0.0 );
//    ocp.subjectTo( AT_END, MarkerArchetTete(x1) - MarkerViolon(x1) == 0.0 );
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


//    ocp.subjectTo(MarkerViolon(x1)(2) - MarkerArchetTete(x1)(2) <= 0.0);
//    ocp.subjectTo(MarkerViolon(x2)(2) - MarkerArchetTete(x2)(2) <= 0.0);

//    ocp.subjectTo(MarkerArchetPoucette(x1)(2) - MarkerViolon(x1)(2) <= 0.0);
//    ocp.subjectTo(MarkerArchetPoucette(x2)(2) - MarkerViolon(x2)(2) <= 0.0);

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
            u_init(i, j) = 0.2;
        }
        for(unsigned int j=nMus; j<nMus+nTau; ++j){
            u_init(i, j) = 0.01;
        }
        for(unsigned int j=nMus+nTau; j<(2*nMus)+nTau; ++j){
            u_init(i, j) = 0.2;
        }
        for(unsigned int j=(2*nMus)+nTau; j<2*(nMus+nTau); ++j){
            u_init(i, j) = 0.01;
        }
    }
    algorithm.initializeControls(u_init);

    VariablesGrid x_init(2*(nQ+nQdot), Grid(t_Start, t_End, 2));

//    // poucette sur COM
//    x_init(0, 0) = 0.1000001;
//    x_init(0, 1) = 0.1000001;
//    x_init(0, 2) = 1.0946872;
//    x_init(0, 3) = 1.5707965;
//    x_init(0, 4) = 1.0564277;
//    x_init(0, 5) = 1.0607269;
//    x_init(0, 6) = -1.725867;

//    // bouton sur COM
//    x_init(1, 0) = -0.39269915;
//    x_init(1, 1) = -0.27353444;
//    x_init(1, 2) = -0.05670261;
//    x_init(1, 3) = 0.439974729;
//    x_init(1, 4) = 0.511486204;
//    x_init(1, 5) = 1.929967317;
//    x_init(1, 6) = -3.35089080;

//    // bouton sur COM
//    x_init(0, nQ+nQdot) = -0.39269915;
//    x_init(0, 1+nQ+nQdot) = -0.27353444;
//    x_init(0, 2+nQ+nQdot) = -0.05670261;
//    x_init(0, 3+nQ+nQdot) = 0.439974729;
//    x_init(0, 4+nQ+nQdot) = 0.511486204;
//    x_init(0, 5+nQ+nQdot) = 1.929967317;
//    x_init(0, 6+nQ+nQdot) = -3.35089080;

//    // poucette sur COM
//    x_init(1, nQ+nQdot) = 0.1000001;
//    x_init(1, 1+nQ+nQdot) = 0.1000001;
//    x_init(1, 2+nQ+nQdot) = 1.09468721;
//    x_init(1, 3+nQ+nQdot) = 1.57079651;
//    x_init(1, 4+nQ+nQdot) = 1.05642775;
//    x_init(1, 5+nQ+nQdot) = 1.06072698;
//    x_init(1, 6+nQ+nQdot) = -1.7258677;

    //COM sur COM :
    x_init(0, 0) = 0.0990382;
    x_init(0, 1) = -0.3329108;
    x_init(0, 2) = 0.63740231;
    x_init(0, 3) = 0.71742303;
    x_init(0, 4) = 0.79172476;
    x_init(0, 5) = 1.26416757;
    x_init(0, 6) = -0.6445338;

    x_init(1, 0) = 0.0990382;
    x_init(1, 1) = -0.3329108;
    x_init(1, 2) = 0.63740231;
    x_init(1, 3) = 0.71742303;
    x_init(1, 4) = 0.79172476;
    x_init(1, 5) = 1.26416757;
    x_init(1, 6) = -0.6445338;

    x_init(0, nQ+nQdot) = 0.0990382382;
    x_init(0, 1+nQ+nQdot) = -0.3329108;
    x_init(0, 2+nQ+nQdot) = 0.63740231;
    x_init(0, 3+nQ+nQdot) = 0.71742303;
    x_init(0, 4+nQ+nQdot) = 0.79172476;
    x_init(0, 5+nQ+nQdot) = 1.26416757;
    x_init(0, 6+nQ+nQdot) = -0.6445338;

    x_init(1, nQ+nQdot) = 0.0990382382;
    x_init(1, 1+nQ+nQdot) = -0.3329108;
    x_init(1, 2+nQ+nQdot) = 0.63740231;
    x_init(1, 3+nQ+nQdot) = 0.71742303;
    x_init(1, 4+nQ+nQdot) = 0.79172476;
    x_init(1, 5+nQ+nQdot) = 1.26416757;
    x_init(1, 6+nQ+nQdot) = -0.6445338;



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
