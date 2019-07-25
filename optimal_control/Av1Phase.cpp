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
unsigned int nMus(0);   // muscles number

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
    Control                 u1("", nMus+nTau, 1);                 //  the  control input  u
    IntermediateState       is1(nQ + nQdot + nMus + nTau); // + 1);

    for (unsigned int i = 0; i < nQ; ++i){
        is1(i) = x1(i);
    }
    for (unsigned int i = 0; i < nQdot; ++i){
        is1(i+nQ) = x1(i+nQ);
    }
    for (unsigned int i = 0; i < nMus; ++i){
        is1(i+nQ+nQdot) = u1(i);
    }
    for (unsigned int i = 0; i < nTau; ++i){
        is1(i+nQ+nQdot+nMus) = u1(i+nMus);
    }

    /* ----------- DEFINE OCP ------------- */
    OCP ocp(t_Start, t_End, nPoints);

    CFunction lagrange(1, lagrangeResidualTorques);
    ocp.minimizeLagrangeTerm( lagrange(is1));

    /* ------------ CONSTRAINTS ----------- */
    CFunction F( nQ+nQdot, forwardDynamicsFromTorqueContact);

    DifferentialEquation    f ;

    (f << dot(x1)) == F(is1);

    ocp.subjectTo(f);

    //Position constraints
    int tagArchetPoucette = 16;
    int tagArchetCOM = 17;
    int tagArchetTete = 18;
    int tagViolon = 34;
    CFunction markerArchetPoucette(3, markerPosition);
    markerArchetPoucette.setUserData((void*) &tagArchetPoucette);
    CFunction markerArchetCOM(3, markerPosition);
    markerArchetCOM.setUserData((void*) &tagArchetCOM);
    CFunction markerArchetTete(3, markerPosition);
    markerArchetTete.setUserData((void*) &tagArchetTete);
    CFunction markerViolon(3, markerPosition);
    markerViolon.setUserData((void*) &tagViolon);

    ocp.subjectTo( AT_START, markerArchetPoucette(x1) - markerViolon(x1) == 0.0 );
//    ocp.subjectTo( AT_START, markerArchetCOM(x1) - markerViolon(x1) == 0.0 );
//    ocp.subjectTo( AT_END, markerArchetCOM(x1) - markerViolon(x1) == 0.0 );
    ocp.subjectTo( AT_END, markerArchetTete(x1) - markerViolon(x1) == 0.0 );

    //Controls constraints
    for (unsigned int i=0; i<nMus; ++i){
         ocp.subjectTo(0.01 <= u1(i) <= 1);
    }

    for (unsigned int i=nMus; i<nMus+nTau; ++i){
         ocp.subjectTo(-100 <= u1(i) <= 100);
    }

    ocp.subjectTo(-PI/8 <= x1(0) <= 0.1);
    ocp.subjectTo(-PI/2 <= x1(1) <= 0.1);
    ocp.subjectTo(-PI/4 <= x1(2) <= PI);
    ocp.subjectTo(-PI/2 <= x1(3) <= PI/2);
    ocp.subjectTo(-0.1 <= x1(4) <= PI);

//    //Contact force constraints
//    CFunction contactforce(2, forceConstraint);
//    double coeffFriction(0.3);
//    ocp.subjectTo(contactforce(is1)(0) - coeffFriction*contactforce(is1)(1) <= 0); // composante selon x: mu*y mu=0.3 d'après www.tangentex.com/CordeViolon.htm
//    ocp.subjectTo(contactforce(is1)(0) + coeffFriction*contactforce(is1)(1) >= 0); // composante selon x: mu*y mu=0.3 d'après www.tangentex.com/CordeViolon.htm
//    ocp.subjectTo(5 <= contactforce(is1)(1) <= 15); // composante selon y; normale au plan du mouvement

    /* ---------- OPTIMIZATION  ------------ */
    OptimizationAlgorithm  algorithm( ocp ) ;       //  construct optimization  algorithm ,
    algorithm.set(MAX_NUM_ITERATIONS, 1000);
    algorithm.set(INTEGRATOR_TYPE, INT_RK45);
    algorithm.set(HESSIAN_APPROXIMATION, FULL_BFGS_UPDATE);
    algorithm.set(KKT_TOLERANCE, 1e-4);

    /* ---------- INITIAL SOLUTION ---------- */
    VariablesGrid u_init(nTau + nMus, Grid(t_Start, t_End, 2));
    for(unsigned int i=0; i<2; ++i){
        for(unsigned int j=0; j<nMus; ++j){
            u_init(i, j) = 0.2;
        }
        for(unsigned int j=nMus; j<nMus+nTau; ++j){
            u_init(i, j) = 0.01;
        }
    }
    algorithm.initializeControls(u_init);

    VariablesGrid x_init(nQ+nQdot, Grid(t_Start, t_End, 2));

    // poucette sur COM
    x_init(0, 0) = 0.1000001;
    x_init(0, 1) = 0.1000001;
    x_init(0, 2) = 1.0946872;
    x_init(0, 3) = 1.5707965;
    x_init(0, 4) = 1.0564277;
    x_init(0, 5) = 1.0607269;
    x_init(0, 6) = -1.725867;

    // bouton sur COM
    x_init(1, 0) = -0.39269915;
    x_init(1, 1) = -0.27353444;
    x_init(1, 2) = -0.05670261;
    x_init(1, 3) = 0.439974729;
    x_init(1, 4) = 0.511486204;
    x_init(1, 5) = 1.929967317;
    x_init(1, 6) = -3.35089080;

//    //COM sur COM :
//    x_init(0, 0) = 0.0990382;
//    x_init(0, 1) = -0.3329108;
//    x_init(0, 2) = 0.63740231;
//    x_init(0, 3) = 0.71742303;
//    x_init(0, 4) = 0.79172476;
//    x_init(0, 5) = 1.26416757;
//    x_init(0, 6) = -0.6445338;

//    x_init(1, 0) = 0.0990382;
//    x_init(1, 1) = -0.3329108;
//    x_init(1, 2) = 0.63740231;
//    x_init(1, 3) = 0.71742303;
//    x_init(1, 4) = 0.79172476;
//    x_init(1, 5) = 1.26416757;
//    x_init(1, 6) = -0.6445338;

    for(unsigned int i=nQ; i<nQ+nQdot; ++i){
         x_init(0, i) = 0.01;
         x_init(1, i) = 0.01;

    }
    algorithm.initializeDifferentialStates(x_init);

    /* ---------- SOLVING THE PROBLEM ---------- */
    algorithm.solve();

    /* ---------- STORING THE RESULTS ---------- */
    createTreePath(resultsPath);
    algorithm.getDifferentialStates((resultsPath + "StatesAv1Phases.txt").c_str());
    algorithm.getControls((resultsPath + "ControlsAv1Phases.txt").c_str());

    end=clock();
    time_exec = double(end - start)/CLOCKS_PER_SEC;
    std::cout<<"Execution time: "<<time_exec<<std::endl;

    return 0;
}
