// C++ (and CasADi) from here on
#include <casadi/casadi.hpp>
#include "casadi/core/optistack.hpp"

#include "forward_dynamics_casadi.h"
#include "BiorbdModel.h"
extern biorbd::Model m;
biorbd::Model m("../../models/simple.bioMod");

int main(){
    // PREPARING THE PROBLEM
    double T(5.0);
    int N(30);
    int nQ(static_cast<int>(m.nbQ()));
    int nu(static_cast<int>(m.nbMuscleTotal() + m.nbGeneralizedTorque()));
    double dt = T/static_cast<double>(N); // length of a control interval
    casadi::Opti opti;
    casadi::MX x(opti.variable(N+1, nQ));
    casadi::MX v(opti.variable(N+1, nQ));
    casadi::MX u(opti.variable(N, nu));

    casadi::Dict opts;
    opts["enable_fd"] = true; // This is for now, someday, it will provide the dynamic derivative!
    casadi::Function f = casadi::external(libforward_dynamics_casadi_name(), opts);

    // OBJECTIVE FUNCTIONS
    casadi::MX obj(0);
    for (int j=0; j<nQ; ++j){
        for (int i=0; i<N; ++i){
            obj += u(i, j) * u(i, j);
        }
    }
    opti.minimize(obj);


    // CONSTRAINTS
    // Continuity constraints (rk4)
    std::vector<casadi::MX> var;
    var.push_back(casadi::MX(1, nQ*2));
    var.push_back(casadi::MX(1, nu));

//    // RK45
//    // Create an integrator (CVodes)
//    for(int i=0; i<N; ++i){
//        for (int j=0; j<nQ; ++j){
//            var[0](j) = x(i, j);
//            var[0](j+nQ) = v(i, j);
//            var[1](j) = u(i, j);
//        }
//        casadi::MXDict ode({{"x", vertcat(x, v)}, {"p", u}, {"ode", f(var)[0] }});
//        casadi::Function F = casadi::integrator("integrator", "cvodes", ode, {{"t0", 0}, {"tf", T/nQ}});
//        // Create an evaluation node
////        std::vector<casadi::MX> I_out = F({x(i), v(i), u(i)});

////        opti.subject_to(x(i+1)==I_out[0]); // close the gaps
////        opti.subject_to(v(i+1)==I_out[1]); // close the gaps
//    }

    // RK4
    for (int i=0; i<N; ++i){ // loop over control intervals
        for (int j=0; j<nQ; ++j){var[1](j) = u(i, j);}
        // Runge-Kutta 4 integration
        for (int j=0; j<nQ; ++j){var[0](j) = x(i, j)           ; var[0](j+nQ) = v(i, j)              ;} casadi::MX k1 = f(var)[0];
        for (int j=0; j<nQ; ++j){var[0](j) = x(i, j)+dt/2*k1(j); var[0](j+nQ) = v(i, j)+dt/2*k1(j+nQ);} casadi::MX k2 = f(var)[0];
        for (int j=0; j<nQ; ++j){var[0](j) = x(i, j)+dt/2*k2(j); var[0](j+nQ) = v(i, j)+dt/2*k2(j+nQ);} casadi::MX k3 = f(var)[0];
        for (int j=0; j<nQ; ++j){var[0](j) = x(i, j)+dt  *k3(j); var[0](j+nQ) = v(i, j)+dt  *k3(j+nQ);} casadi::MX k4 = f(var)[0];
        for (int j=0; j<nQ; ++j){
            casadi::MX x_next = x(i, j) + dt/6 * (k1(j) + 2*k2(j) + 2*k3(j) + k4(j));
            casadi::MX v_next = v(i, j) + dt/6 * (k1(j+nQ) + 2*k2(j+nQ) + 2*k3(j+nQ) + k4(j+nQ));
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

////// VANILLA EOCAR
////int main(){
////// Declare variables
////SX p = SX::sym("p"); // position
////SX v = SX::sym("v"); // velocity
////SX u = SX::sym("u"); // control
////SX x = vertcat(p,v);

////// Number of differential states
////int nx = static_cast<int>(x.size1());

////// Number of controls
////int nu = static_cast<int>(u.size1());

////// Bounds and initial guess for the state
////vector<double> x0_min = {   0,    0 };
////vector<double> x0_max = {   0,    0 };
////vector<double> x_min  = {-500, -500 };
////vector<double> x_max  = { 500,  500 };
////vector<double> xf_min = { 100,    0 };
////vector<double> xf_max = { 100,    0 };
////vector<double> x_init = {   0,    0 };

////// Bounds and initial guess for the control
////vector<double> u_min =  { -100 };
////vector<double> u_max  = {  100 };
////vector<double> u_init = {  0.0 };


////// Final time
////double tf = 5.0;

////// Number of shooting nodes
////int ns = 30;

////// ODE right hand side and quadrature
////SXDict ode = {{"x", x}, {"p", u}, {"ode", vertcat(v, u)}};

////// Create an integrator (CVodes)
////Function F = integrator("integrator", "cvodes", ode, {{"t0", 0}, {"tf", tf/ns}});

////// Total number of NLP variables
////int NV = nx*(ns+1) + nu*ns;

////// Declare variable vector for the NLP
////MX V = MX::sym("V",NV);

////// NLP variable bounds and initial guess
////vector<double> v_min,v_max,v_init;

////// Offset in V
////int offset=0;

////// State at each shooting node and control for each shooting interval
////vector<MX> X, U;
////for(int k=0; k<ns; ++k){
////  // Local state
////  X.push_back( V.nz(Slice(offset,offset+nx)));
////  if(k==0){
////    v_min.insert(v_min.end(), x0_min.begin(), x0_min.end());
////    v_max.insert(v_max.end(), x0_max.begin(), x0_max.end());
////  } else {
////    v_min.insert(v_min.end(), x_min.begin(), x_min.end());
////    v_max.insert(v_max.end(), x_max.begin(), x_max.end());
////  }
////  v_init.insert(v_init.end(), x_init.begin(), x_init.end());
////  offset += nx;

////  // Local control
////  U.push_back( V.nz(Slice(offset,offset+nu)));
////  v_min.insert(v_min.end(), u_min.begin(), u_min.end());
////  v_max.insert(v_max.end(), u_max.begin(), u_max.end());
////  v_init.insert(v_init.end(), u_init.begin(), u_init.end());
////  offset += nu;
////}

////// State at end
////X.push_back(V.nz(Slice(offset,offset+nx)));
////v_min.insert(v_min.end(), xf_min.begin(), xf_min.end());
////v_max.insert(v_max.end(), xf_max.begin(), xf_max.end());
////v_init.insert(v_init.end(), x_init.begin(), x_init.end());
////offset += nx;

////// Make sure that the size of the variable vector is consistent with the number of variables that we have referenced
////casadi_assert(offset==NV, "");

////// Objective function
////MX J = 0;

//////Constraint function and bounds
////vector<MX> g;

////// Loop over shooting nodes
////for(unsigned int k=0; k<static_cast<unsigned int>(ns); ++k){
////  // Create an evaluation node
////  MXDict I_out = F(MXDict{{"x0", X[k]}, {"p", U[k]}});

////  // Save continuity constraints
////  g.push_back( I_out.at("xf") - X[k+1] );

////  // Add objective function contribution
////  J += I_out.at("qf");
////}

////// NLP
////MXDict nlp = {{"x", V}, {"f", J}, {"g", vertcat(g)}};

////// Set options
////Dict opts;
////opts["ipopt.tol"] = 1e-6;
////opts["ipopt.max_iter"] = 100;

////// Create an NLP solver and buffers
////Function solver = nlpsol("nlpsol", "ipopt", nlp, opts);
////std::map<std::string, DM> arg, res;

////// Bounds and initial guess
////arg["lbx"] = v_min;
////arg["ubx"] = v_max;
////arg["lbg"] = 0;
////arg["ubg"] = 0;
////arg["x0"] = v_init;

////// Solve the problem
////res = solver(arg);

////// Optimal solution of the NLP
////vector<double> V_opt(res.at("x"));

////// Get the optimal state trajectory
////vector<double> r_opt(ns+1), s_opt(ns+1);
////for(int i=0; i<=ns; ++i){
////  r_opt[i] = V_opt.at(i*(nx+1));
////  s_opt[i] = V_opt.at(1+i*(nx+1));
////}
////cout << "r_opt = " << endl << r_opt << endl;
////cout << "s_opt = " << endl << s_opt << endl;

////// Get the optimal control
////vector<double> u_opt(ns);
////for(int i=0; i<ns; ++i){
////  u_opt[i] = V_opt.at(nx + i*(nx+1));
////}
////cout << "u_opt = " << endl << u_opt << endl;


////return 0;
////}
