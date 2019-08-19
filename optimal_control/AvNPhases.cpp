#include <acado_optimal_control.hpp>
#include <bindings/acado_gnuplot/gnuplot_window.hpp>
#include "includes/dynamics.h"
#include "includes/objectives.h"
#include "includes/constraints.h"
#include "includes/utils.h"
#include <time.h>

#ifndef PI
#define PI 3.141592
#endif

using namespace std;
USING_NAMESPACE_ACADO

biorbd::Model m("../../models/BrasViolon.bioMod");

unsigned int nQ(m.nbQ());               // states number
unsigned int nQdot(m.nbQdot());         // derived states number
unsigned int nTau(m.nbGeneralizedTorque());           // controls number
unsigned int nTags(m.nTags());          // markers number
unsigned int nMus(m.nbMuscleTotal());   // muscles number
unsigned int nPhases(2);
GeneralizedCoordinates Q(nQ), Qdot(nQdot), Qddot(nQdot);
GeneralizedTorque Tau(nTau);
std::vector<biorbd::muscles::StateDynamics> state(nMus); // controls

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
    if (nPhases % 2 != 0)
        throw runtime_error("nPhases must be an even number");
    std::cout << "nb de phases: " << nPhases << std::endl;

    /* ---------- INITIALIZATION ---------- */
    DifferentialState       x("", nPhases*(nQ+nQdot), 1);
    Control                 u("", nPhases*(nMus+nTau), 1);
    IntermediateState       is("", nPhases*(nQ+nQdot+nMus+nTau), 1);




    for (unsigned int i=0; i< nPhases; i++){
        for (unsigned int j = 0; j < nQ+nQdot; ++j){
            is(j+(i*(nQ+nQdot+nMus+nTau))) = x(j+(i*(nQ+nQdot)));
        }
        for (unsigned int j = 0; j < nMus+nTau; ++j){
            is(j+nQ+nQdot+(i*(nQ+nQdot+nMus+nTau))) = u(j+(i*(nMus+nTau)));
        }
    }

    /* ----------- DEFINE OCP ------------- */
    OCP ocp(t_Start, t_End, nPoints);

    CFunction lagrange(1, lagrangeResidualTorquesMultistage);
    ocp.minimizeLagrangeTerm(lagrange(u));

    /* ------------ CONSTRAINTS ----------- */
    CFunction F( nPhases*(nQ+nQdot), forwardDynamicsMultiStage);
    DifferentialEquation    f ;
    (f << dot(x)) == F(is);
    ocp.subjectTo(f);

    //Position constraints
    int tagArchetPoucette = 16;
    int tagArchetCOM = 17;
    int tagArchetTete = 18;
    int tagViolon = 34;
    CFunction markerArchetPoucette(3, markerPosition);
    markerArchetPoucette.setUserData(static_cast<void*>(&tagArchetPoucette));
    CFunction markerArchetCOM(3, markerPosition);
    markerArchetCOM.setUserData(static_cast<void*>(&tagArchetCOM));
    CFunction markerArchetTete(3, markerPosition);
    markerArchetTete.setUserData(static_cast<void*>(&tagArchetTete));
    CFunction markerViolon(3, markerPosition);
    markerViolon.setUserData(static_cast<void*>(&tagViolon));

    ocp.subjectTo( AT_START, markerArchetPoucette(x) - markerViolon(x) == 0.0 );
    ocp.subjectTo( AT_END, markerArchetTete(x) - markerViolon(x) == 0.0 );

//    for(unsigned int i=0; i<nPhases-1; ++i){
//        for(unsigned int j=0; j< nQ+nQdot; ++j){
//            //ocp.subjectTo( 0.0, x(( ((i+1) % 2)*(nQ+nQdot))+j), -x((i*(nQ+nQdot))+j), 0.0 );
//            ocp.subjectTo( 0.0, x(((i+1)*(nQ+nQdot))+j), -x((i*(nQ+nQdot))+j), 0.0 );
//            ocp.subjectTo( 0.0, x((i*(nQ+nQdot))+j), -x(((i+1)*(nQ+nQdot))+j), 0.0 );
//        }
//    }

    for(unsigned int j=0; j<nQ+nQdot; ++j){
        ocp.subjectTo( 0.0, x(nQ+nQdot + j), -x(j), 0.0 );
        ocp.subjectTo( 0.0, x(j), -x(nQ+nQdot + j), 0.0 );
    }

    //Controls constraints
    for (unsigned int i=0; i<nPhases; ++i){
        for (unsigned int j=0; j<nMus; ++j){
             ocp.subjectTo(0.01 <= u( i*(nMus+nTau) + j ) <= 1);
        }
    }

    for (unsigned int i=0; i<nPhases; ++i){
        for (unsigned int j=0; j<nTau; ++j){
             ocp.subjectTo(-100 <= u( i*(nMus+nTau) + nMus + j ) <= 100);
        }
    }

    for (unsigned int i=0; i<nPhases; ++i){
        ocp.subjectTo(-PI/8 <= x(i*(nQ+nQdot)+0) <= 0.1);
        ocp.subjectTo(-PI/2 <= x(i*(nQ+nQdot)+1) <= 0.1);
        ocp.subjectTo(-PI/4 <= x(i*(nQ+nQdot)+2) <= PI);
        ocp.subjectTo(-PI/2 <= x(i*(nQ+nQdot)+3) <= PI/2);
        ocp.subjectTo(-0.1  <= x(i*(nQ+nQdot)+4) <= PI);
        ocp.subjectTo(-PI   <= x(i*(nQ+nQdot)+5) <= PI);
        ocp.subjectTo(-PI   <= x(i*(nQ+nQdot)+6) <= PI);
    }
    for (unsigned int i=0; i<nPhases; ++i){
        for (unsigned int q=0; q<nQ; ++q)
            ocp.subjectTo(-50 <= x(i*(nQ+nQdot)+nQ+q) <= 50);
    }

    /* ---------- OPTIMIZATION  ------------ */
    OptimizationAlgorithm  algorithm( ocp ) ;       //  construct optimization  algorithm ,
    algorithm.set(MAX_NUM_ITERATIONS, 1000);
    algorithm.set(INTEGRATOR_TYPE, INT_RK45);
    algorithm.set(HESSIAN_APPROXIMATION, FULL_BFGS_UPDATE);
    algorithm.set(KKT_TOLERANCE, 1e-4);

    /* ---------- INITIAL SOLUTION ---------- */
    VariablesGrid u_init(nPhases*(nTau + nMus), Grid(t_Start, t_End, 2));
    for(unsigned int i=0; i<nPhases; ++i){
        for(unsigned int j=0; j<nMus; ++j){
            u_init(0, i*(nMus+nTau) + j ) = 0.2;
            u_init(1, i*(nMus+nTau) + j ) = 0.2;
        }
        for(unsigned int j=0; j<nTau; ++j){
            u_init(0, i*(nMus+nTau) + nMus + j ) = 0.01;
            u_init(1, i*(nMus+nTau) + nMus + j ) = 0.01;
        }
    }
    algorithm.initializeControls(u_init);

    VariablesGrid x_init(nPhases*(nQ+nQdot), Grid(t_Start, t_End, 2));
    for(unsigned int i=0; i<nPhases/2; ++i){
        // poucette sur COM
        x_init(0, 2*i*(nQ+nQdot)+0) = 0.1000001;
        x_init(0, 2*i*(nQ+nQdot)+1) = 0.1000001;
        x_init(0, 2*i*(nQ+nQdot)+2) = 1.0946872;
        x_init(0, 2*i*(nQ+nQdot)+3) = 1.5707965;
        x_init(0, 2*i*(nQ+nQdot)+4) = 1.0564277;
        x_init(0, 2*i*(nQ+nQdot)+5) = 1.0607269;
        x_init(0, 2*i*(nQ+nQdot)+6) = -1.725867;

        // bouton sur COM
        x_init(1, 2*i*(nQ+nQdot)+0) = -0.39269915;
        x_init(1, 2*i*(nQ+nQdot)+1) = -0.27353444;
        x_init(1, 2*i*(nQ+nQdot)+2) = -0.05670261;
        x_init(1, 2*i*(nQ+nQdot)+3) = 0.439974729;
        x_init(1, 2*i*(nQ+nQdot)+4) = 0.511486204;
        x_init(1, 2*i*(nQ+nQdot)+5) = 1.929967317;
        x_init(1, 2*i*(nQ+nQdot)+6) = -3.35089080;

        // bouton sur COM
        x_init(0, ((2*i)+1)*(nQ+nQdot)+0) = -0.39269915;
        x_init(0, ((2*i)+1)*(nQ+nQdot)+1) = -0.27353444;
        x_init(0, ((2*i)+1)*(nQ+nQdot)+2) = -0.05670261;
        x_init(0, ((2*i)+1)*(nQ+nQdot)+3) = 0.439974729;
        x_init(0, ((2*i)+1)*(nQ+nQdot)+4) = 0.511486204;
        x_init(0, ((2*i)+1)*(nQ+nQdot)+5) = 1.929967317;
        x_init(0, ((2*i)+1)*(nQ+nQdot)+6) = -3.35089080;

        // poucette sur COM
        x_init(1, ((2*i)+1)*(nQ+nQdot)+0) = 0.1000001;
        x_init(1, ((2*i)+1)*(nQ+nQdot)+1) = 0.1000001;
        x_init(1, ((2*i)+1)*(nQ+nQdot)+2) = 1.09468721;
        x_init(1, ((2*i)+1)*(nQ+nQdot)+3) = 1.57079651;
        x_init(1, ((2*i)+1)*(nQ+nQdot)+4) = 1.05642775;
        x_init(1, ((2*i)+1)*(nQ+nQdot)+5) = 1.06072698;
        x_init(1, ((2*i)+1)*(nQ+nQdot)+6) = -1.7258677;
    }

    for(unsigned int i=0; i<nPhases; ++i){
        for(unsigned int j=0; j<nQdot; ++j){
             x_init(0, i*(nQ+nQdot) + nQ + j) = 0.01;
             x_init(1, i*(nQ+nQdot) + nQ + j) = 0.01;
        }
    }
    algorithm.initializeDifferentialStates(x_init);

    /* ---------- SOLVING THE PROBLEM ---------- */
    algorithm.solve();

    /* ---------- STORING THE RESULTS ---------- */
    createTreePath(resultsPath);
    algorithm.getDifferentialStates((resultsPath + "StatesAvNPhases.txt").c_str());
    algorithm.getControls((resultsPath + "ControlsAvNPhases.txt").c_str());

    end=clock();
    time_exec = double(end - start)/CLOCKS_PER_SEC;
    std::cout<<"Execution time: "<<time_exec<<std::endl;

    return 0;
}