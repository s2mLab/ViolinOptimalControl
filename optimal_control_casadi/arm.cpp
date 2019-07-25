// C++ (and CasADi) from here on
#include <casadi/casadi.hpp>
#include "casadi/core/optistack.hpp"

#include "forward_dynamics_casadi.h"
#include "s2mMusculoSkeletalModel.h"
extern s2mMusculoSkeletalModel m;
s2mMusculoSkeletalModel m("../../models/simple.bioMod");

int main(){
    // PREPARING THE PROBLEM
    double tf(5.0);
    int ns(30);
    double dt = tf/static_cast<double>(ns); // length of a control interval

    // Differential variables
    casadi::MX u = casadi::MX::sym("u1");
    casadi::MX p = casadi::MX::sym("p"), v = casadi::MX::sym("v");
    casadi::MX x(vertcat(p, v));

    // Number of differential states
    int nQ = static_cast<int>(x.size1());
    int nU = static_cast<int>(u.size1());

    // Bounds and initial guess for the control
    std::vector<double> u_min =  { -100 };
    std::vector<double> u_max  = {  100  };
    std::vector<double> u_init = {  0  };

    // Bounds and initial guess for the state
    std::vector<double> x0_min = { 0, 0 };
    std::vector<double> x0_max = { 0, 0 };
    std::vector<double> x_min  = { -500, -500 };
    std::vector<double> x_max  = { 500, 500 };
    std::vector<double> xf_min = { 100, 0 };
    std::vector<double> xf_max = { 100, 0 };
    std::vector<double> x_init = { 0, 0 };

    casadi::Dict opts_dyn;
    opts_dyn["enable_fd"] = true; // This is for now, someday, it will provide the dynamic derivative!
    casadi::Function f = casadi::external(libforward_dynamics_casadi_name(), opts_dyn);

    // ODE right hand side
    casadi::MXDict ode = {{"x", x}, {"p", u}, {"ode", f(std::vector<casadi::MX>({x, u}))[0]}};
    casadi::Function F( casadi::integrator("integrator", "cvodes", ode, {{"t0", 0}, {"tf", dt}} ) );

    // Total number of NLP variables
    int NV = nQ*(ns+1) + nU*ns;

    // Declare variable vector for the NLP
    casadi::MX V = casadi::MX::sym("V",NV);

    // NLP variable bounds and initial guess
    std::vector<double> v_min,v_max,v_init;

    // Offset in V
    int offset=0;

    // State at each shooting node and control for each shooting interval
    std::vector<casadi::MX> X, U;
    for(int k=0; k<ns; ++k){
      // Local state
      X.push_back( V.nz(casadi::Slice(offset,offset+nQ)));
      if(k==0){
        v_min.insert(v_min.end(), x0_min.begin(), x0_min.end());
        v_max.insert(v_max.end(), x0_max.begin(), x0_max.end());
      } else {
        v_min.insert(v_min.end(), x_min.begin(), x_min.end());
        v_max.insert(v_max.end(), x_max.begin(), x_max.end());
      }
      v_init.insert(v_init.end(), x_init.begin(), x_init.end());
      offset += nQ;

      // Local control
      U.push_back( V.nz(casadi::Slice(offset,offset+nU)));
      v_min.insert(v_min.end(), u_min.begin(), u_min.end());
      v_max.insert(v_max.end(), u_max.begin(), u_max.end());
      v_init.insert(v_init.end(), u_init.begin(), u_init.end());
      offset += nU;
    }

    // State at end
    X.push_back(V.nz(casadi::Slice(offset,offset+nQ)));
    v_min.insert(v_min.end(), xf_min.begin(), xf_min.end());
    v_max.insert(v_max.end(), xf_max.begin(), xf_max.end());
    v_init.insert(v_init.end(), x_init.begin(), x_init.end());
    offset += nQ;

    // Make sure that the size of the variable vector is consistent with the number of variables that we have referenced
    casadi_assert(offset==NV, "");

    // Objective function
    casadi::MX J = 0;

    //Constraint function and bounds
    std::vector<casadi::MX> g;

    // Loop over shooting nodes
    for(unsigned int k=0; k<static_cast<unsigned int>(ns); ++k){
      // Create an evaluation node
      casadi::MXDict I_out = F(casadi::MXDict{{"x0", X[k]}, {"p", U[k]}});

      // Save continuity constraints
      g.push_back( I_out.at("xf") - X[k+1] );

      // Add objective function contribut`ion
      J += U[k] * U[k];
    }

    // NLP
    casadi::MXDict nlp = {{"x", V}, {"f", J}, {"g", vertcat(g)}};

    // Set options
    casadi::Dict opts;
    opts["ipopt.tol"] = 1e-5;
    opts["ipopt.max_iter"] = 100;

    // Create an NLP solver and buffers
    casadi::Function solver = nlpsol("nlpsol", "ipopt", nlp, opts);
    std::map<std::string, casadi::DM> arg, res;

    // Bounds and initial guess
    arg["lbx"] = v_min;
    arg["ubx"] = v_max;
    arg["lbg"] = 0;
    arg["ubg"] = 0;
    arg["x0"] = v_init;

    // Solve the problem
    res = solver(arg);

    // Optimal solution of the NLP
    std::vector<double> V_opt(res.at("x"));

    // Get the optimal state trajectory
    std::vector<double> r_opt(ns+1), s_opt(ns+1);
    for(int i=0; i<=ns; ++i){
      r_opt[i] = V_opt.at(i*(nQ+1));
      s_opt[i] = V_opt.at(1+i*(nQ+1));
    }
    std::cout << "r_opt = " << std::endl << r_opt << std::endl;
    std::cout << "s_opt = " << std::endl << s_opt << std::endl;

    // Get the optimal control
    std::vector<double> u_opt(ns);
    for(int i=0; i<ns; ++i){
      u_opt[i] = V_opt.at(nQ + i*(nQ+1));
    }
    std::cout << "u_opt = " << std::endl << u_opt << std::endl;


    return 0;
}
