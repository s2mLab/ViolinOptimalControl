#include <casadi/casadi.hpp>
#include "casadi/core/optistack.hpp"

using namespace std;

int main(){

    // INITIALIZATION
    double T(10.0);
    int N(30);
    int d(1);           // number of dimensions, here d = ns - 1
    int ns(2);          // s for State(s), e.g. for a value of 2 : position x and speed v
    int nu(1);          // one control : acceleration
    double dt = T/static_cast<double>(N);    // time step

    casadi::Opti opti;
    casadi::MX s(opti.variable(2*d,N+1)); // s for states : vertical concatenation of positions x and speeds v, size = 2d*(N+1)
    casadi::MX u(opti.variable(nu, N));

    // OBJECTIVE FUNCTION
    casadi::MX obj(0);
    for (int i=0; i<nu; i++)
    {
        for (int j=0; j<N; j++)
        {
            obj += u(i, j) * u(i, j) * dt;          // dt = cte, on pourrait l'enlever ça changerait le résultat ? A test
        }
    }
    opti.minimize(obj);                             // The optimizer is prepared to minimize this objective function

    // DYNAMIC DEFINITION
    casadi::MX s_sym = casadi::MX::sym("s", 2*d,1);
    casadi::MX u_sym = casadi::MX::sym("u", nu, 1);
    casadi::Function dyn = casadi::Function("DynamicFunction",          // Don't put a space in the name otherwise bug
                       {s_sym, u_sym},
                       {vertcat(s_sym(casadi::Slice(d,2*d),0), u_sym)},
                       {"s", "u"}, {"sdot"});

    // RK4
    casadi::MXDict var2;
    std::vector<casadi::MX> var;
    var.push_back(casadi::MX(2*d,1));
    var.push_back(casadi::MX(nu,1));

    for (int j=0; j<N; j++)
    {
        var2["u"] = u(casadi::Slice(0, nu), j);
        var[1] = u(casadi::Slice(0, nu), j);

        var2["s"] = s(casadi::Slice(0,2*d), j);
        var[0] = s(casadi::Slice(0,2*d), j);
        casadi::MX k1(dyn(var)[0]);
        casadi::MX k12(dyn(var2).at("sdot"));
        std::cout << k1 << std::endl;
        std::cout << k12 << std::endl;

        var[0] = s(casadi::Slice(0,2*d),j) + (dt/2)*k1;
        casadi::MX k2(dyn(var)[0]);

        var[0] = s(casadi::Slice(0,2*d),j) + (dt/2)*k2;
        casadi::MX k3(dyn(var)[0]);

        var[0] = s(casadi::Slice(0,2*d),j) + dt*k3;
        casadi::MX k4(dyn(var)[0]);

        casadi::MX s_next(s(casadi::Slice(0,2*d),j) + (dt/6)*(k1+2*k2 + 2*k3 + k4));


        //  Continuity conditions
        opti.subject_to(s(casadi::Slice(0,2*d),j+1) == s_next);

    }

    // CONSTRAINTS (continuity constraints, boundary conditions, path constraints)

        //  Continuity conditions --> inside RK4

        // Boundary conditions
        opti.subject_to(s(casadi::Slice(0,2*d), 0) == 0);
        //opti.subject_to(s(1, 0) == 0);
        opti.subject_to(s(0, N) == 10);
        opti.subject_to(s(1, N) == 0);          // Need to be complete if d>1

        // Path constraints

        for (int j=0; j<N+1; j++)
            {
                opti.subject_to(s(0, j) >= -10);        // Need to be complete if d>1
                opti.subject_to(s(0, j) <= 20);
                opti.subject_to(s(1, j) >= -100);
                opti.subject_to(s(1, j) <= 100);
            }
        for (int j=0; j<N; j++)
        {
            opti.subject_to(u(casadi::Slice(0,nu), j) >= -100);
            opti.subject_to(u(casadi::Slice(0,nu), j) <= 100);
        }

    // OPTIMIZATION
    opti.solver("ipopt");
    casadi::OptiSol sol = opti.solve();

    // DISPLAYING RESULTS
    cout << endl << sol.value(s) << endl << endl;
    cout << sol.value(u) << endl << endl;

    cout << "There are indeed 92 variables in play: (N+1=)31 of positions, 31 of speeds and (N=)30 of controls (lines 17, 18 and 20)."
 << endl << endl;
    cout << "There are also 64 equality constraints: 30*2 for continuity of positions & speeds + 4 for boundary conditions (lines 76-89)." << endl << endl;
    cout << "Finally, there are indeed 184 unequal constraints: 31*4 + 30*2 (cf. path constraints lines 91)." << endl << endl;

    return 0;

}
