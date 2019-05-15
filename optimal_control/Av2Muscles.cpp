#include <acado_optimal_control.hpp>
#include <bindings/acado_gnuplot/gnuplot_window.hpp>
#include <s2mMusculoSkeletalModel.h>
#include <vector>

using namespace std;
USING_NAMESPACE_ACADO

/* ---------- Model ---------- */

s2mMusculoSkeletalModel m("../../models/Bras.pyoMod");


static int nQ(static_cast<int>(m.nbQ()));               // states number
static int nQdot(static_cast<int>(m.nbQdot()));         // derived states number
static int nTau(static_cast<int>(m.nbTau()));           // torque number
static int nTags(static_cast<int>(m.nTags()));          // markers number
static int nMus(static_cast<int>(m.nbMuscleTotal()));   // muscles number

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
}

#define  NOL   nMus                 // number of lagrange objective functions
void myLagrangeObjectiveFunction( double *x, double *g, void *user_data){
    for(int i=0; i<nMus; ++i)
        g[i] = x[i+nQ+nQdot];
}


#define  NOM   1                 // number of mayer objective functions
void myMayerObjectiveFunction( double *x, double *g, void *user_data){
    g[0] = (x[0]-PI/2)*(x[0]-PI/2);
    //g[1] = (x[1]-PI/6)*(x[1]-PI/6);
}

#define  NI   nQ+nQdot                 // number of initial value constraints
void myInitialValueConstraint( double *x, double *g, void *user_data){
    for(int i=0; i<nQ; ++i){
        g[i]=x[i]-0.01;
        g[i+nQ]=x[i+nQ];
    }
}

#define  NE   1                 // number of end-point / terminal constraints
void myEndPointConstraint( double *x, double *g, void *user_data){
    g[0]=x[0]-PI/2;
}


int  main ()
{
    std::cout << "nb de marqueurs: " << nTags << std::endl<< std::endl;
    std::cout << "nb de muscles: " << nMus << std::endl<< std::endl;

    /* ---------- INITIALIZATION ---------- */
    Parameter               T;                              //  the  time  horizon T
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
    OCP ocp( t_Start, T , nPoints);

    CFunction Mayer( NOM, myMayerObjectiveFunction);
    CFunction Lagrange( NOL, myLagrangeObjectiveFunction);
    ocp.minimizeMayerTerm( T );
    ocp.minimizeLagrangeTerm( Lagrange(is) );

    /* ------------ CONSTRAINTS ----------- */
    DifferentialEquation    f(0.0, T) ;
    CFunction F( NX, fowardDynamics);
    CFunction I( NI, myInitialValueConstraint   );
    CFunction E( NE, myEndPointConstraint       );

    ocp.subjectTo( (f << dot(x)) == F(is) );                          //  differential  equation,
    ocp.subjectTo( AT_START, I(is) ==  0.0 );
    ocp.subjectTo( AT_END  , E(is) ==  0.0 );
    ocp.subjectTo(0.01 <= u <= 1);

    ocp.subjectTo(3.0 <= T <= 10.0);

    /* ---------- OPTIMIZATION  ------------ */
    OptimizationAlgorithm  algorithm( ocp ) ;       //  construct optimization  algorithm ,

    VariablesGrid u_init(nMus, Grid(t_Start, t_End, 2));
    VariablesGrid x_init(2, Grid(t_Start, t_End, 2));

    for(int i=0; i<2; ++i){
        for(int j=0; j<nMus; ++j){
            u_init(i, j) = 0.1;
        }
    }

    for(int i=0; i<nQ; ++i){
         x_init(0, i) = 0.1;
         x_init(1, i) = 0.1;
    }
    for(int i=nQ; i<nQdot; ++i){
         x_init(0, i) = 0.0;
         x_init(1, i) = 0.0;
    }

    algorithm.initializeControls(u_init);
    algorithm.initializeDifferentialStates(x_init);

    GnuplotWindow window;                           //  visualize  the  results  in  a  Gnuplot  window
    window.addSubplot(  x ,  "STATES x" ) ;
    window.addSubplot( u ,  "CONTROL  u" ) ;
    algorithm << window;
    algorithm.solve();                              //  solve the problem .

    algorithm.getDifferentialStates("../Results/StatesAv2Muscles.txt");
    algorithm.getControls("../Results/ControlsAv2Muscles.txt");

    return 0;
}


