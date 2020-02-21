// C++ (and CasADi) from here on
#include <casadi/casadi.hpp>
#include "casadi/core/optistack.hpp"

#include "biorbd.h"
extern biorbd::Model m;
biorbd::Model m("../../models/simple.bioMod");

biorbd::utils::Vector Fd(
        biorbd::Model& model,
        const casadi::MX& states,
        const casadi::MX& controls)
{
    biorbd::rigidbody::GeneralizedCoordinates Q;
    biorbd::rigidbody::GeneralizedVelocity QDot;
    biorbd::rigidbody::GeneralizedAcceleration QDDot(model.nbQ());
    biorbd::rigidbody::GeneralizedTorque Tau;

    Q = states(casadi::Slice(0, static_cast<casadi_int>(model.nbQ())));
    QDot = states(casadi::Slice(static_cast<casadi_int>(model.nbQ()),
                                static_cast<casadi_int>(model.nbQ()*2)));
    Tau = controls(casadi::Slice(0, static_cast<casadi_int>(model.nbQ())));

    RigidBodyDynamics::ForwardDynamics(model, Q, QDot, Tau, QDDot);
    return vertcat(QDot, QDDot);
}

int main(){
    // PREPARING THE PROBLEM
    double T(5.0);
    int N(30);
    int nQ(static_cast<int>(m.nbQ()));
    int nu(static_cast<int>(m.nbGeneralizedTorque()));
    double dt = T/static_cast<double>(N); // length of a control interval
    casadi::Opti opti;
    casadi::MX x(opti.variable(N+1, nQ));
    casadi::MX v(opti.variable(N+1, nQ));
    casadi::MX u(opti.variable(N, nu));

    // Prepare the dynamic function
    casadi::MX states = casadi::MX::sym("x", m.nbQ()*2, 1);
    casadi::MX controls = casadi::MX::sym("p", m.nbQ(), 1);
    casadi::Function f = casadi::Function( "ForwardDyn",
                                {states, controls},
                                {Fd(m, states, controls)},
                                {"states", "controls"},
                                {"statesdot"}).expand();

    // OBJECTIVE FUNCTIONS
    casadi::MX obj(0);
    for (int j=0; j<nQ; ++j){
        for (int i=0; i<N; ++i){
            obj += u(i, j) * u(i, j);
        }
    }
    opti.minimize(obj);


    // CONSTRAINTS
    // Continuity constraints
    std::vector<casadi::MX> var;
    var.push_back(casadi::MX(nQ*2, 1));
    var.push_back(casadi::MX(nu, 1));

    // RK4
    for (int i=0; i<N; ++i){ // loop over control intervals
        for (int j=0; j<nQ; ++j){
            var[1](j) = u(i, j);}

        // Runge-Kutta 4 integration
        for (int j=0; j<nQ; ++j){
            var[0](j) = x(i, j);
            var[0](j+nQ) = v(i, j);
        }
        casadi::MX k1 = f(var)[0];
        for (int j=0; j<nQ; ++j){
            var[0](j) = x(i, j)+k1(j)*(dt/2);
            var[0](j+nQ) = v(i, j)+k1(j+nQ)*(dt/2);
        }
        casadi::MX k2 = f(var)[0];
        for (int j=0; j<nQ; ++j){
            var[0](j) = x(i, j)+k2(j)*(dt/2);
            var[0](j+nQ) = v(i, j)+k2(j+nQ)*(dt/2);
        }
        casadi::MX k3 = f(var)[0];
        for (int j=0; j<nQ; ++j){
            var[0](j) = x(i, j)+k3(j)*dt;
            var[0](j+nQ) = v(i, j)+k3(j+nQ)*dt;
        }
        casadi::MX k4 = f(var)[0];
        for (int j=0; j<nQ; ++j){
            casadi::MX x_next = x(i, j) + (k1(j) + k2(j)*2 + k3(j)*2 + k4(j))*(dt/6);
            casadi::MX v_next = v(i, j) + (k1(j+nQ) + k2(j+nQ)*2 + k3(j+nQ)*2 + k4(j+nQ))*(dt/6);
            opti.subject_to(x(i+1, j)==x_next); // close the gaps
            opti.subject_to(v(i+1, j)==v_next); // close the gaps
        }
    }

    // Boundary conditions
    opti.subject_to(x(N, 0) == 10);
    opti.subject_to(x(N, 1) == 10);
    opti.subject_to(x(N, 2) == 10);
    opti.subject_to(x(N, 3) == 1);
    opti.subject_to(x(N, 4) == 1);
    opti.subject_to(x(N, 5) == 1);
    opti.subject_to(x(N, 6) == 1);
    opti.subject_to(x(N, 7) == 1);
    opti.subject_to(x(N, 8) == 1);
    for (int j=0; j<nQ; ++j){
        opti.subject_to(x(0, j) == 0);
        opti.subject_to(v(0, j) == 0);
        opti.subject_to(v(N, j) == 0);

        // Path constraints
        for (int i=0; i<N+1; ++i){
            opti.subject_to(x(i, j) >= -500);
            opti.subject_to(x(i, j) <= 500);
            opti.subject_to(v(i, j) >= -100);
            opti.subject_to(v(i, j) <= 100);
        }
        for (int i=0; i<N; ++i){
            opti.subject_to(u(i, j) >= -100);
            opti.subject_to(u(i, j) <= 100);
        }
    }

    // INITIAL GUESS

    // SOLVING
    opti.solver("ipopt");
    casadi::OptiSol sol = opti.solve();

    // SHOWING THE RESULTS
    std::cout << sol.value(x) << std::endl << std::endl;
    std::cout << sol.value(v) << std::endl << std::endl;
    std::cout << sol.value(u) << std::endl << std::endl;

    return 0;
}
