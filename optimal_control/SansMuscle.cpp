#include <acado_optimal_control.hpp>
#include <bindings/acado_gnuplot/gnuplot_window.hpp>
#include <s2mMusculoSkeletalModel.h>

using namespace std;
USING_NAMESPACE_ACADO

/* ---------- Model ---------- */

s2mMusculoSkeletalModel m("../Modeles/ModeleSansMuscle.bioMod");


static int nQ(static_cast<int>(m.nbQ()));               // states number
static int nQdot(static_cast<int>(m.nbQdot()));         // derived states number
static int nTau(static_cast<int>(m.nbTau()));           // controls number
static int nTags(static_cast<int>(m.nTags()));          // markers number


const double t_Start=0.0;
const double t_End= 10.0;
const int nPoints(30);


/* ---------- Functions ---------- */

#define  NX   nQ + nQdot        // number of differential states
void fowardDynamics( double *x, double *rhs, void *user_data){
    s2mGenCoord Q(static_cast<unsigned int>(nQ));           // states
    s2mGenCoord Qdot(static_cast<unsigned int>(nQdot));     // derivated states
    s2mTau Tau(static_cast<unsigned int>(nTau));            // controls
    s2mGenCoord Qddot(static_cast<unsigned int>(nQdot));

    for (int i = 0; i<nQ; ++i){
        Q[i] = x[i];
        Qdot[i] = x[i+nQ];
        Tau[i] = x[i+nQ+nQdot];
    }

    RigidBodyDynamics::ForwardDynamics(m, Q, Qdot, Tau, Qddot);

    for (int i = 0; i<nQ; ++i){
       rhs[i] = Qdot[i];
       rhs[i + nQdot] = Qddot[i];
   }
/*   Q[0]=1.5637;
   Qdot[0]=4.2277e-01;
   Tau[0]=-1.0949e+01;
   RigidBodyDynamics::ForwardDynamics(m, Q, Qdot, Tau, Qddot);

   std::cout << Qddot <<endl;
*/
}

#define  NOL   1                 // number of lagrange objective functions
void myLagrangeObjectiveFunction( double *x, double *g, void *user_data ){
    g[0] = 0;
}

#define  NOM   1                 // number of mayer objective functions
void myMayerObjectiveFunction( double *x, double *g, void *user_data ){
    double obj = x[0]-PI/2;
    g[0] = obj*obj;
}

#define  NI   2                 // number of initial value constraints
void myInitialValueConstraint( double *x, double *g, void *user_data ){
    for (int i =0; i<nQ + nQdot; ++i) {
        g[i] =  x[i];
    }
}

#define  NE   2                 // number of end-point / terminal constraints
void myEndPointConstraint( double *x, double *g, void *user_data ){
    g[0]=x[0]-PI/2;                         // rotation de 90°
    g[0]=x[1];                              // vitesse nulle
}


int  main ()
{
    /* ---------- INITIALIZATION ---------- */
    Parameter               T;                              //  the  time  horizon T
    DifferentialState       x("",nQ+nQdot,1);               //  the  differential states
    Control                 u("", nTau, 1);          //  the  control input  u
    IntermediateState       is(nQ + nQdot + nTau);

    for (int i = 0; i < nQ; ++i)
        is(i) = x(i);                   //*scalingQ(i);
    for (int i = 0; i < nQdot; ++i)
        is(i+nQ) = x(i+nQ);             //*scalingQdot(i);
    for (int i = 0; i < nTau; ++i)
        is(i+nQ+nQdot) = u(i);          //*scalingQdot(i);

    /* ----------- DEFINE OCP ------------- */
    CFunction Mayer( NOM, myMayerObjectiveFunction);
    CFunction Lagrange( NOL, myLagrangeObjectiveFunction);
    OCP ocp( t_Start, T , nPoints);                        // time  horizon
    ocp.minimizeMayerTerm( Mayer(is) );                    // Mayer term
    ocp.minimizeLagrangeTerm( Lagrange(is) );                    // Lagrange term


    /* ------------ CONSTRAINTS ----------- */
    DifferentialEquation    f;                             //  the  differential  equation
    CFunction F( NX, fowardDynamics);
    CFunction I( NI, myInitialValueConstraint   );
    CFunction E( 1, myEndPointConstraint       );

    ocp.subjectTo( (f << dot(x)) == F(is) );                          //  differential  equation,
    ocp.subjectTo( AT_START, I(is) ==  0.0 );
    ocp.subjectTo( AT_END  , E(is) ==  0.0 );
    ocp.subjectTo(-100 <= u <= 100);



    /* ---------- OPTIMIZATION  ------------ */
    OptimizationAlgorithm  algorithm( ocp ) ;       //  construct optimization  algorithm ,

    VariablesGrid u_init(1, Grid(t_Start, t_End, 2));
    u_init(0, 0) = 0.1;
    u_init(0, 1) = 0.1;
    algorithm.initializeControls(u_init);

    VariablesGrid x_init(2, Grid(t_Start, t_End, 2));
    x_init(0, 0) = 0.1;
    x_init(0, 1) = 0.1;
    x_init(1, 0) = 0.1;
    x_init(1, 1) = 0.1;
    algorithm.initializeDifferentialStates(x_init);

    GnuplotWindow window;                           //  visualize  the  results  in  a  Gnuplot  window
    window.addSubplot(  x ,  "DISTANCE x" ) ;
    window.addSubplot( u ,  "CONTROL  u" ) ;
    algorithm << window;
    algorithm.solve();                              //  and solve the problem .

    algorithm.getDifferentialStates("../Resultats/StatesSansMuscle.txt");
    algorithm.getControls("../Resultats/ControlsSansMuscle.txt");

    return 0;
}

