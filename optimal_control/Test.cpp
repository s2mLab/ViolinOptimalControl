#include "biorbd.h"
#include "includes/integrator.h"


#include "includes/biorbd_declarer.h"
biorbd::Model m("../../models/BrasViolon.bioMod");
#include "includes/biorbd_initializer.h"

int main (){

    double x[nQ + nQdot] = {1.0000019017476415e-01,	1.0000019174575936e-01,
            1.0974854998234509e+00,	1.5707965169214431e+00,	1.0563643477850255e+00,
            1.0688691915911472e+00,	-1.7268961079933003e+00, 4.6793742849957243e-01,
            -1.6931300137553904e-01, 3.5465470083533002e-01, -4.1754770738803781e-01,
            1.4074901166006851e-01, 3.0698605734625894e-01, 1.5308241019830612e-01};

    for(unsigned int i = 0; i < nQ; ++i)
        Q[i] = x[i];
    for(unsigned int i = 0; i < nQdot; ++i)
        Qdot[i] = x[i+nQ];


    double torques[nTau]  = {9.9998080119966769e-03, 7.1216513064073284e-01,
                             3.5511034562935351e-01, 7.9053458721196712e-01,
                             1.2090013854635781e-01, 1.1448031133796523e-01,
                             1.7257617394280274e-02};

    double muscles[nMus] = {1.0851990037012542e-01,	5.1416849841112063e-02,
                            2.7187839682557424e-01,	8.4264952706869484e-02,
                            3.8003490622410749e-02,	7.7659357280003982e-01,
                            4.7952511822163077e-02,	2.4078193944695614e-01,
                            2.1325799299084292e-01,	4.2999334698826486e-01,
                            3.4699406211489404e-02,	1.3541669437403208e+00,
                            -5.6832945602240885e-02, -4.5450359867909171e-02,
                            -9.2912327192316430e-02, 2.4475037529961306e-02,
                            1.2510245012433752e-01,	5.8915912054233111e-01};

    biorbd::utils::Vector controls(nMus+nTau);


    for(unsigned int i = 0; i < nMus; ++i)
        controls(i) = muscles[i];
    for(unsigned int i = 0; i < nTau; ++i)
        controls(i+nMus) = torques[i];




    AcadoIntegrator integrator(m);
    integrator.integrateKinematics(Q, Qdot, controls, 0, 1.6129032258064516e-02, 1.6129032258064516e-03);

    biorbd::rigidbody::GeneralizedCoordinates QOut(m),QdotOut(m);
    for (unsigned int i=0; i<integrator.nbInterationStep(); ++i) {
        integrator.getIntegratedKinematics(i, QOut, QdotOut);
        std::cout << "Q = " << QOut.transpose() << std::endl;
        std::cout << "Qdot = " << QdotOut.transpose() << std::endl << std::endl;
  }


//    for(unsigned int i = 0; i < nQ + nQdot; ++i)
//        states[i] = x[i];

}


