#include <acado_optimal_control.hpp>
#include <bindings/acado_gnuplot/gnuplot_window.hpp>
#include <s2mMusculoSkeletalModel.h>

//#define DebugForce ;
//#define DebugActivation ;
//#define DebugLongueur ;
using namespace std;
USING_NAMESPACE_ACADO

/* ---------- Model ---------- */

s2mMusculoSkeletalModel m("../Modeles/ModeleAv1Muscle.bioMod");


static int nQ(static_cast<int>(m.nbQ()));               // states number
static int nQdot(static_cast<int>(m.nbQdot()));         // derived states number
static int nTau(static_cast<int>(m.nbTau()));           // torque number
static int nTags(static_cast<int>(m.nTags()));          // markers number
static int nMus(static_cast<int>(m.nbMuscleTotal()));          // muscle number

const double t_Start=0.0;
const double t_End= 10.0;
const int nPoints(30);

/* ---------- Functions ---------- */

#define  NX   nQ + nQdot        // number of differential states

void fowardDynamics( double *x, double *rhs, void *user_data){
    s2mGenCoord Q(static_cast<unsigned int>(nQ));           // states
    s2mGenCoord Qdot(static_cast<unsigned int>(nQdot));     // derivated states

    for (int i = 0; i<nQ; ++i){
        Q[i] = x[i];
        Qdot[i] = x[i+nQ];
    }
    m.updateMuscles(m, Q, Qdot, true);
    std::vector<s2mMuscleStateActual> state;// controls
    for (int i = 0; i<nMus; ++i)
        state.push_back(s2mMuscleStateActual(0, x[i+nQ+nQdot]));

     //Calcul de torque
    s2mTau Tau = m.muscularJointTorque(m, state, true, &Q, &Qdot);

    //Fonction de dynamique directe
    s2mGenCoord Qddot(static_cast<unsigned int>(nQdot));
    RigidBodyDynamics::ForwardDynamics(m, Q, Qdot, Tau, Qddot);

    for (int i = 0; i<nQ; ++i){
        rhs[i] = Qdot[i];
        rhs[i + nQdot] = Qddot[i];
    }

#ifdef DebugForce
    for(int i=0; i<m.nbTau(); ++i){
        if (Tau[i]>1e5){
            std::vector<int> L;
            for(int j=0; j<nMus; ++j){
                if (m.musclesLengthJacobian(m, true, &Q).coeff(j,i)!=0){
                    L.push_back(j);
                }
            }
         std::cout << "Torque "<<i<<" is too high" << endl;
         std::cout << "Check the optimal lenth, the maximal force or the tendon slack lenth of muscles " << L <<endl;
         L.clear();
         }
    }
#endif

#ifdef DebugActivation
    for(int i=0; i<m.nbTau(); ++i){
        if (Tau[i]>0.1){
            int c=0;
            std::vector<int> L;
            for(int j=0; j<nMus; ++j){
                if (m.musclesLengthJacobian(m, true, &Q).coeff(j,i)!=0){
                    L.push_back(j);
                    if(state[j].activation()>0.015)
                        c++;
                }
            }
            if (c=0)
                std::cout << "Passive force of the muscles " <<L<< " is too high. Check the tendon slack lenth."<<endl;
         }
    }
#endif

#ifdef DebugLongueur
    for(int i=0; i<m.nbMuscleGroups(); ++i){
        for(int j=0; j<m.muscleGroup(i).nbMuscles(); ++j)
            if (m.muscleGroup(i).muscle(j).get()->length(m, Q) <= 0)
                std::cout << "La longueur du muscle " << i << " est inférieur á 0" <<endl;
    }
#endif
}

#define  NOL   1                 // number of lagrange objective functions
void myLagrangeObjectiveFunction( double *x, double *g, void *user_data ){
    g[0] = x[2];
}


#define  NOM   1                 // number of mayer objective functions
void myMayerObjectiveFunction( double *x, double *g, void *user_data ){
    double obj = x[0]-PI/2;
    g[0] = obj*obj;
}

#define  NI   2                 // number of initial value constraints
void myInitialValueConstraint( double *x, double *g, void *user_data ){
    g[0] = x[0]-0.01;
    g[1] = x[1];

}

#define  NE   1                 // number of end-point / terminal constraints
void myEndPointConstraint( double *x, double *g, void *user_data ){
   // g[0]=x[0]-PI/2;                         // rotation de 90°
    g[0]=x[1];                              // vitesse nulle

}


int  main ()
{
    std::cout << "nb de marqueurs: " << nTags << std::endl<< std::endl;
    std::cout << "nb de muscles: " << nMus << std::endl<< std::endl;

    /* ---------- INITIALIZATION ---------- */
   // Parameter               T;                              //  the  time  horizon T
    DifferentialState       x("",nQ+nQdot,1);               //  the  differential states
    Control                 u("", nMus, 1);                 //  the  control input  u
    IntermediateState       is(nQ + nQdot + nMus);

    for (int i = 0; i < nQ; ++i)
        is(i) = x(i);
    for (int i = 0; i < nQdot; ++i)
        is(i+nQ) = x(i+nQ);
    for (int i = 0; i < nMus; ++i)
        is(i+nQ+nQdot) = u(i);

    /* ----------- DEFINE OCP ------------- */
    OCP ocp( t_Start, t_End , nPoints);

    CFunction Mayer( NOM, myMayerObjectiveFunction);
    CFunction Lagrange( NOL, myLagrangeObjectiveFunction);
    ocp.minimizeMayerTerm( Mayer(is) );
    ocp.minimizeLagrangeTerm( Lagrange(is) );

    /* ------------ CONSTRAINTS ----------- */
    DifferentialEquation    f ;
    CFunction F( NX, fowardDynamics);
    CFunction I( NI, myInitialValueConstraint   );
    CFunction E( NE, myEndPointConstraint       );

    ocp.subjectTo( (f << dot(x)) == F(is) );                          //  differential  equation,
    ocp.subjectTo( AT_START, I(is) ==  0.0 );
    ocp.subjectTo( AT_END  , E(is) ==  0.0 );
    ocp.subjectTo(0.01 <= u <= 1);

    /* ---------- OPTIMIZATION  ------------ */
    OptimizationAlgorithm  algorithm( ocp ) ;       //  construct optimization  algorithm ,

    algorithm.initializeDifferentialStates("../Initialisation/X1Muscle.txt");
    //algorithm.initializeParameters("../Initialisation/T1Muscle.txt");
    algorithm.initializeControls("../Initialisation/U1Muscle.txt");


    GnuplotWindow window;                           //  visualize  the  results  in  a  Gnuplot  window
    window.addSubplot(  x ,  "STATES x" ) ;
    window.addSubplot( u ,  "CONTROL  u" ) ;
    algorithm << window;
    algorithm.solve();                              //  solve the problem .

    algorithm.getDifferentialStates("../Resultats/StatesAv1Muscle.txt");
    //algorithm.getParameters("../Resultats/ParametresAv1Muscle.txt");
    algorithm.getControls("../Resultats/ControlsAv1Muscle.txt");

    return 0;
}


