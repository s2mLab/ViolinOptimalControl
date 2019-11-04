#include <acado_optimal_control.hpp>
#include <memory>
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
unsigned int nTau(m.nbGeneralizedTorque());           // torques number
unsigned int nMarkers(m.nMarkers());          // markers number
unsigned int nMus(m.nbMuscleTotal());   // muscles number
unsigned int nPhases(2);

GeneralizedCoordinates Q(nQ), Qdot(nQdot), Qddot(nQdot);
GeneralizedTorque Tau(nTau);
std::vector<std::shared_ptr<biorbd::muscles::StateDynamics>> musclesStates(nMus);

static int tagArchetPoucette = 16;
static int tagArchetCOM = 17;
static int tagArchetTete = 18;
static int tagViolon = 34;

static int idxSegmentArchet = 8;


const double t_Start = 0.0;
const double t_End = 0.5;
const int nPoints(31);

int  main ()
{
    for (unsigned int i=0; i<nMus; ++i)
        musclesStates[i] = std::make_shared<biorbd::muscles::StateDynamics>(biorbd::muscles::StateDynamics());

    std::string resultsPath("../Results/");

    clock_t start,end;
    double time_exec;
    start=clock();

    std::cout << "nb de muscles: " << nMus << std::endl;
    std::cout << "nb de torques: " << nTau << std::endl;
    std::cout << "nb de marqueurs: " << nMarkers << std::endl;

    /* ---------- INITIALIZATION ---------- */
    std::vector<DifferentialState> x;
    std::vector<Control> u;
    std::vector<IntermediateState> is;

    /* ----------- DEFINE OCP ------------- */
    OCP ocp(t_Start, t_End, nPoints);
    CFunction lagrangeRT(1, lagrangeResidualTorques);
    CFunction lagrangeA(1, lagrangeActivations);
    CFunction F( nQ+nQdot, forwardDynamicsFromMuscleActivationAndTorque);
    DifferentialEquation f ;

    //Position constraints
    CFunction markerArchetPoucette(3, markerPosition);
    markerArchetPoucette.setUserData(static_cast<void*>(&tagArchetPoucette));
    CFunction markerArchetCOM(3, markerPosition);
    markerArchetCOM.setUserData(static_cast<void*>(&tagArchetCOM));
    CFunction markerArchetTete(3, markerPosition);
    markerArchetTete.setUserData(static_cast<void*>(&tagArchetTete));
    CFunction markerViolon(3, markerPosition);
    markerViolon.setUserData(static_cast<void*>(&tagViolon));

    CFunction projeteArchet(2, orthogonalProjected);
    int idxMarkers[2] = {tagViolon, idxSegmentArchet};
    projeteArchet.setUserData(static_cast<void*>(idxMarkers));


    for (unsigned int p=0; p<nPhases; ++p){
        x.push_back(DifferentialState("",nQ+nQdot,1));
        u.push_back(Control("", nMus+nTau, 1));
        is.push_back(IntermediateState(nQ + nQdot + nMus + nTau));

        for (unsigned int i = 0; i < nQ; ++i)
            is[p](i) = x[p](i);
        for (unsigned int i = 0; i < nQdot; ++i)
            is[p](i+nQ) = x[p](i+nQ);
        for (unsigned int i = 0; i < nMus; ++i)
            is[p](i+nQ+nQdot) = u[p](i);
        for (unsigned int i = 0; i < nTau; ++i)
            is[p](i+nQ+nQdot+nMus) = u[p](i+nMus);

        /* ------------ CONSTRAINTS ----------- */
        (f << dot(x[p])) == F(is[p]);

        for (int i = 1; i < nPoints-1; ++i)
         ocp.subjectTo(i, projeteArchet(x[p]) == 0.0);

        if(p==0){

            ocp.subjectTo( AT_START, markerArchetPoucette(x[p]) - markerViolon(x[p]) == 0.0 );
            ocp.subjectTo( AT_END, markerArchetTete(x[p]) - markerViolon(x[p]) == 0.0 );
        }
        else{
            ocp.subjectTo( 0.0, x[p], -x[p-1], 0.0 );
            ocp.subjectTo( 0.0, x[p-1], -x[p], 0.0 );
        }

        //Controls constraints
        for (unsigned int i=0; i<nMus; ++i)
             ocp.subjectTo(0.01 <= u[p](i) <= 1);
        for (unsigned int i=0; i<nTau; ++i)
             ocp.subjectTo(-100 <= u[p](nMus+i) <= 100);

        // path constraints
        ocp.subjectTo(-PI/8 <= x[p](0) <= 0.1);
        ocp.subjectTo(-PI/2 <= x[p](1) <= 0.1);
        ocp.subjectTo(-PI/4 <= x[p](2) <= PI);
        ocp.subjectTo(-PI/2 <= x[p](3) <= PI/2);
        ocp.subjectTo(-0.1  <= x[p](4) <= PI);
        ocp.subjectTo(-PI   <= x[p](5) <= PI);
        ocp.subjectTo(-PI   <= x[p](6) <= PI);

        for (unsigned int j=0; j<nQdot; ++j)
            ocp.subjectTo(-50 <= x[p](nQ + j) <= 50);

}
    ocp.subjectTo(f);

    /* ------------ OBJECTIVE ----------- */
    Expression sumLagrange = lagrangeRT(u[0])+ lagrangeA(u[0]);
    for(unsigned int p=1; p<nPhases; ++p)
        sumLagrange += lagrangeRT(u[p]) + lagrangeA(u[p]);
    ocp.minimizeLagrangeTerm( sumLagrange ); // WARNING

    /* ---------- OPTIMIZATION  ------------ */
    OptimizationAlgorithm  algorithm( ocp ) ;
    algorithm.set(MAX_NUM_ITERATIONS, 1000);
    algorithm.set(INTEGRATOR_TYPE, INT_RK45);
    algorithm.set(HESSIAN_APPROXIMATION, FULL_BFGS_UPDATE);
    algorithm.set(KKT_TOLERANCE, 1e-4);

    ACADO::returnValue n = algorithm.initializeDifferentialStates((resultsPath + "InitStates.txt").c_str(), BT_TRUE);
    n.~returnValue();
    ACADO::returnValue n2 = algorithm.initializeControls((resultsPath + "InitControls.txt").c_str());

    /* ---------- INITIAL SOLUTION ---------- */
//    VariablesGrid u_init(nPhases*(nTau + nMus), Grid(t_Start, t_End, 2));
//    for(unsigned int i=0; i<nPhases; ++i){
//        for(unsigned int j=0; j<nMus; ++j){
//            u_init(0, i*(nMus+nTau) + j ) = 0.2;
//            u_init(1, i*(nMus+nTau) + j ) = 0.2;
//        }
//        for(unsigned int j=0; j<nTau; ++j){
//            u_init(0, i*(nMus+nTau) + nMus + j ) = 0.01;
//            u_init(1, i*(nMus+nTau) + nMus + j ) = 0.01;
//        }
//    }
//    algorithm.initializeControls(u_init);

//    VariablesGrid x_init(nPhases*(nQ+nQdot), Grid(t_Start, t_End, 2));
//    for(unsigned int i=0; i<nPhases/2; ++i){
//        /* Solution INITIALE Camille */

//                // poucette sur COM
//                x_init(0, 2*i*(nQ+nQdot)+0) = 0.09973;
//                x_init(0, 2*i*(nQ+nQdot)+1) = 0.09733;
//                x_init(0, 2*i*(nQ+nQdot)+2) = 1.05710;
//                x_init(0, 2*i*(nQ+nQdot)+3) = 1.56950;
//                x_init(0, 2*i*(nQ+nQdot)+4) = 1.07125;
//                x_init(0, 2*i*(nQ+nQdot)+5) = 0.95871;
//                x_init(0, 2*i*(nQ+nQdot)+6) = -1.7687;

//                // bouton sur COM
//                x_init(1, 2*i*(nQ+nQdot)+0) = -0.39107;
//                x_init(1, 2*i*(nQ+nQdot)+1) = -0.495383;
//                x_init(1, 2*i*(nQ+nQdot)+2) = -0.089030;
//                x_init(1, 2*i*(nQ+nQdot)+3) = 0.1485315;
//                x_init(1, 2*i*(nQ+nQdot)+4) = 0.8569764;
//                x_init(1, 2*i*(nQ+nQdot)+5) = 1.9126840;
//                x_init(1, 2*i*(nQ+nQdot)+6) = -0.490220;

//                // bouton sur COM
//                x_init(0, ((2*i)+1)*(nQ+nQdot)+0) = -0.39107;
//                x_init(0, ((2*i)+1)*(nQ+nQdot)+1) = -0.495383;
//                x_init(0, ((2*i)+1)*(nQ+nQdot)+2) = -0.089030;
//                x_init(0, ((2*i)+1)*(nQ+nQdot)+3) = 0.1485315;
//                x_init(0, ((2*i)+1)*(nQ+nQdot)+4) = 0.8569764;
//                x_init(0, ((2*i)+1)*(nQ+nQdot)+5) = 1.9126840;
//                x_init(0, ((2*i)+1)*(nQ+nQdot)+6) = -0.490220;

//                // poucette sur COM
//                x_init(1, ((2*i)+1)*(nQ+nQdot)+0) = 0.09973;
//                x_init(1, ((2*i)+1)*(nQ+nQdot)+1) = 0.09733;
//                x_init(1, ((2*i)+1)*(nQ+nQdot)+2) = 1.05710;
//                x_init(1, ((2*i)+1)*(nQ+nQdot)+3) = 1.56950;
//                x_init(1, ((2*i)+1)*(nQ+nQdot)+4) = 1.07125;
//                x_init(1, ((2*i)+1)*(nQ+nQdot)+5) = 0.95871;
//                x_init(1, ((2*i)+1)*(nQ+nQdot)+6) = -1.7687;

//    }


//    for(unsigned int i=0; i<nPhases; ++i){
//        for(unsigned int j=0; j<nQdot; ++j){
//             x_init(0, i*(nQ+nQdot) + nQ + j) = 0.01;
//             x_init(1, i*(nQ+nQdot) + nQ + j) = 0.01;
//        }
//    }
//    algorithm.initializeDifferentialStates(x_init);

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
