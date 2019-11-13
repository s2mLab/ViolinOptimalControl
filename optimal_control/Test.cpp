#include "biorbd.h"
#include "includes/integrator.h"


#include "includes/biorbd_declarer.h"
biorbd::Model m("../../models/simple.bioMod");
#include "includes/biorbd_initializer.h"

int main (){
//    double x[14] = {1.0000019017476415e-01,	1.0000019174575936e-01,	1.0974854998234509e+00,	1.5707965169214431e+00,	1.0563643477850255e+00,	1.0688691915911472e+00,	-1.7268961079933003e+00, 4.6793742849957243e-01, -1.6931300137553904e-01, 3.5465470083533002e-01, -4.1754770738803781e-01, 1.4074901166006851e-01, 3.0698605734625894e-01, 1.5308241019830612e-01};
//     double u[7] = {1.3541669437403208e+00,	-5.6832945602240885e-02, -4.5450359867909171e-02, -9.2912327192316430e-02, 2.4475037529961306e-02, 1.2510245012433752e-01,	5.8915912054233111e-01};//    for(unsigned int i=0; i<7; ++i){
//        Q[i] = x[i];
//        Qdot[i] = x[i+7];
//        Tau[i] = u[i];
//    }
//    std::cout << "Q = " << Q.transpose() << std::endl;
//    std::cout << "Qdot = " << Qdot.transpose() << std::endl;
    AcadoIntegrator integrator(m);
    Q.setZero();
    Qdot.setZero();
    Tau.setZero();
    integrator.integrateKinematics(Q, Qdot, Tau, 0, 1, 0.1);
    biorbd::rigidbody::GeneralizedCoordinates QOut(m),QdotOut(m);
    for (unsigned int i=0; i<integrator.nbInterationStep(); ++i) {
        integrator.getIntegratedKinematics(i, QOut, QdotOut);
        std::cout << "Q = " << QOut.transpose() << std::endl;
        std::cout << "Qdot = " << QdotOut.transpose() << std::endl << std::endl;
    }
}


