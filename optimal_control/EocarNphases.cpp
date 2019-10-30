#include <acado_optimal_control.hpp>
#include <bindings/acado_gnuplot/gnuplot_window.hpp>
#include "BiorbdModel.h"
#include "includes/dynamics.h"
#include "includes/objectives.h"
#include "includes/constraints.h"
#include "includes/utils.h"
#include <vector>
#include <memory>

using namespace std;
USING_NAMESPACE_ACADO

/* ---------- Model ---------- */
biorbd::Model m("../../models/ModelTest.bioMod");
unsigned int nQ(m.nbQ());               // states number
unsigned int nQdot(m.nbQdot());         // derived states number
unsigned int nTau(m.nbGeneralizedTorque());           // controls number
unsigned int nMarkers(m.nMarkers());          // markers number
unsigned int nMus(m.nbMuscleTotal());   // muscles number
unsigned int nPhases(2); // phases number
GeneralizedCoordinates Q(nQ), Qdot(nQdot), Qddot(nQdot);
GeneralizedTorque Tau(nTau);
std::vector<std::shared_ptr<biorbd::muscles::StateDynamics>> musclesStates(nMus);

const double t_Start = 0.0;
const double t_End = 4.0;
const int nPoints(31);


int  main ()
{
    for (unsigned int i=0; i<nMus; ++i)
        musclesStates[i] = std::make_shared<biorbd::muscles::StateDynamics>(biorbd::muscles::StateDynamics());

     std::string resultsPath("../Results/");
     std::string resultsPath2 (resultsPath + "InitStatesEocar.txt");


    //    printf( "nQdot vaut : %d \n", nQdot);
//    printf( "nQ vaut : %d \n", nQ);
//    printf( "nTau vaut : %d \n", nTau);
//    printf( "nMarkers vaut : %d \n", nMarkers);
//    printf( "nMus vaut : %d \n", nMus);

    /* ---------- INITIALIZATION ---------- */

//    DifferentialState       x1("", nQ+nQdot, 1);
//    Control                 u1("", nMus+nTau, 1);
//    DifferentialEquation    f;
//    IntermediateState       is1("", nQ+nQdot+nMus+nTau, 1);

    /* ---------- INITIALIZATION ---------- */
    std::vector<DifferentialState> x1;
    std::vector<Control> u1;
    std::vector<IntermediateState> is1;


    for (unsigned int p=0; p<nPhases; ++p){
        x1.push_back(DifferentialState("",nQ+nQdot,1));
        u1.push_back(Control("", nMus+nTau, 1));
        is1.push_back(IntermediateState(nQ + nQdot + nMus + nTau));




        for (unsigned int i = 0; i < nQ + nQdot; ++i) // On remplit le IntermediateState avec les X
        is1[p](i) = x1[p](i);



        for (unsigned int i = 0; i < nMus + nTau; ++i) // Puis avec les controles
        is1[p](i+nQ+nQdot) = u1[p](i);

    }


    //    /* ----------- DEFINE OCP ------------- */
    //    OCP ocp( t_Start , t_End , nPoints-1);
    //    CFunction mayer(1, mayerVelocity); // Dans une sortie, on fais la somme des vitesses au carré
    //    CFunction lagrange(1, lagrangeResidualTorques);
    //    ocp.minimizeLagrangeTerm(lagrange(u1)); // On minimise le terme de Lagrange
    //    ocp.minimizeMayerTerm(mayer(is1)); // On minimise le terme de Mayer

    /* ----------- DEFINE OCP ------------- */
    OCP ocp(t_Start, t_End, nPoints);
    CFunction lagrangeRT(1, lagrangeResidualTorques);
    CFunction lagrangeA(1, lagrangeActivations);
    CFunction F(nQ+nQdot, forwardDynamicsFromJointTorque);
    DifferentialEquation f ;




    /* ------------ CONSTRAINTS ----------- */

    for (unsigned int p=0; p<nPhases; ++p){

        (f << dot( x1[p] )) == F(is1[p]);


        if (p == 0) {

            ocp.subjectTo( AT_START, x1[p](0) ==  0 );
            ocp.subjectTo( AT_END, x1[p](0) ==  10 );

            ocp.subjectTo( AT_START, x1[p](1) ==  0);
            ocp.subjectTo( AT_END, x1[p](1) ==  0);
        }
        else {
            ocp.subjectTo( 0.0, x1[p], -x1[p-1], 0.0 );
            ocp.subjectTo( 0.0, x1[p-1], -x1[p], 0.0 );
        }
        //    ocp.subjectTo( AT_START, x1(2) ==  0 );
        //    ocp.subjectTo( AT_END, x1(2) ==  0 );

        //    ocp.subjectTo( AT_START, x1(3) ==  0 );
        //    ocp.subjectTo( AT_END, x1(3) ==  0 );


        for (unsigned int i=0; i<nMus; ++i)
            ocp.subjectTo(0.01 <= u1[p](i) <= 1);



        for (unsigned int i=nMus; i<nMus+nTau; ++i)
            ocp.subjectTo(-100 <= u1[p](i) <= 100);
   }
    ocp.subjectTo( f ) ;
    /* ------------ OBJECTIVE ----------- */
    Expression sumLagrange = lagrangeRT(u1[0])+ lagrangeA(u1[0]);
    for(unsigned int p=1; p<nPhases; ++p)
        sumLagrange += lagrangeRT(u1[p]) + lagrangeA(u1[p]);

    ocp.minimizeLagrangeTerm( sumLagrange ); // WARNING








    /* ---------- VISUALIZATION ------------ */

    GnuplotWindow window;                           //  visualize  the  results  in  a  Gnuplot  window
    for (unsigned int p=0; p<nPhases; ++p){

        window.addSubplot( x1[p](0), "Etat 0 " );
        window.addSubplot( x1[p](1), "Etat 1 " );
        window.addSubplot( u1[p](0),  "CONTROL  1" ) ;
    }


    /* ---------- OPTIMIZATION  ------------ */

    OptimizationAlgorithm  algorithm( ocp ) ;       //  construct optimization  algorithm ,
    algorithm.set(MAX_NUM_ITERATIONS, 500);
    algorithm.initializeDifferentialStates((resultsPath + "InitStatesEocar.txt").c_str(),BT_TRUE);
    algorithm.initializeControls((resultsPath + "InitControlsEocar.txt").c_str());


    algorithm << window;
    algorithm.solve();



    createTreePath(resultsPath);
    algorithm.getDifferentialStates((resultsPath + "StatesEocar.txt").c_str());
    algorithm.getControls((resultsPath + "ControlsEocar.txt").c_str());


    return  0;
}
