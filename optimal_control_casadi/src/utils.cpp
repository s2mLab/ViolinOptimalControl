#include "utils.h"

void defineDifferentialVariables(
        ProblemSize& ps,
        casadi::MX& u,
        casadi::MX& x)
{
    // Controls
    for (unsigned int i=0; i<m.nbTau(); ++i)
        u = vertcat(u, casadi::MX::sym("Control_" + m.nameDof()[i]));

    // States
    casadi::MX q;
    casadi::MX qdot;
    for (unsigned int i=0; i<m.nbQ(); ++i){
        q = vertcat(q, casadi::MX::sym("Q_" + m.nameDof()[i]));
        qdot = vertcat(qdot, casadi::MX::sym("Qdot_" + m.nameDof()[i]));
    }
    x = vertcat(q, qdot);

    // Number of differential states
    ps.nx = static_cast<unsigned int>(x.size1());
    ps.nu = static_cast<unsigned int>(u.size1());
}

void defineMultipleShootingNodes(
        const ProblemSize& ps,
        const BoundaryConditions &u,
        const BoundaryConditions &x,
        casadi::MX &V,
        BoundaryConditions &v,
        std::vector<casadi::MX> &U,
        std::vector<casadi::MX> &X)
{
    // Total number of NLP variables
    unsigned int NV = ps.nx*(ps.ns+1) + ps.nu*ps.ns;

    // Declare variable vector for the NLP
    V = casadi::MX::sym("V",NV);

    // Offset in V
    int offset=0;

    // State at each shooting node and control for each shooting interval
    for(unsigned int k=0; k<ps.ns; ++k){
        // Local state
        X.push_back( V.nz(casadi::Slice(offset,offset+static_cast<int>(ps.nx))));
        if(k==0){
            v.min.insert(v.min.end(), x.starting_min.begin(), x.starting_min.end());
            v.max.insert(v.max.end(), x.starting_max.begin(), x.starting_max.end());
        } else {
            v.min.insert(v.min.end(), x.min.begin(), x.min.end());
            v.max.insert(v.max.end(), x.max.begin(), x.max.end());
        }
        v.initialGuess.insert(v.initialGuess.end(), x.initialGuess.begin(), x.initialGuess.end());
        offset += ps.nx;

        // Local control
        U.push_back( V.nz(casadi::Slice(offset,offset+static_cast<int>(ps.nu))));
        v.min.insert(v.min.end(), u.min.begin(), u.min.end());
        v.max.insert(v.max.end(), u.max.begin(), u.max.end());
        v.initialGuess.insert(v.initialGuess.end(), u.initialGuess.begin(), u.initialGuess.end());
        offset += ps.nu;
    }

    // State at end
    X.push_back(V.nz(casadi::Slice(offset,offset+static_cast<int>(ps.nx))));
    v.min.insert(v.min.end(), x.end_min.begin(), x.end_min.end());
    v.max.insert(v.max.end(), x.end_max.begin(), x.end_max.end());
    v.initialGuess.insert(v.initialGuess.end(), x.initialGuess.begin(), x.initialGuess.end());
    offset += ps.nx;

    // Make sure that the size of the variable vector is consistent with the number of variables that we have referenced
    casadi_assert(offset==static_cast<int>(NV), "");
}

void continuityConstraints(
        const casadi::Function &dynamics,
        const ProblemSize &ps,
        const std::vector<casadi::MX> &U,
        const std::vector<casadi::MX> &X,
        std::vector<casadi::MX> &g)
{
    // Loop over shooting nodes
    for(unsigned int k=0; k<ps.ns; ++k){
        // Create an evaluation node
        casadi::MXDict I_out = dynamics(casadi::MXDict{{"x0", X[k]}, {"p", U[k]}});

        // Save continuity constraints
        g.push_back( I_out.at("xf") - X[k+1] );
    }
}

void minimizeControls(
        const ProblemSize &ps,
        const std::vector<casadi::MX> &,
        const std::vector<casadi::MX> &U,
        casadi::MX &obj)
{
    obj = 0;
    for(unsigned int k=0; k<ps.ns; ++k)
        obj += casadi::MX::dot(U[k], U[k]);
}

void solveProblemWithIpopt(
        const casadi::MX &V,
        const BoundaryConditions& vBounds,
        const casadi::MX &obj,
        const std::vector<casadi::MX> &constraints,
        std::vector<double>& V_opt)
{
    // NLP
    casadi::MXDict nlp = {{"x", V},
                          {"f", obj},
                          {"g", vertcat(constraints)}};

    // Set options
    casadi::Dict opts;
    opts["ipopt.tol"] = 1e-5;
    opts["ipopt.max_iter"] = 100;

    // Create an NLP solver and buffers
    casadi::Function solver = nlpsol("nlpsol", "ipopt", nlp, opts);
    std::map<std::string, casadi::DM> arg, res;

    // Bounds and initial guess
    arg["lbx"] = vBounds.min;
    arg["ubx"] = vBounds.max;
    arg["lbg"] = 0;
    arg["ubg"] = 0;
    arg["x0"] = vBounds.initialGuess;

    // Solve the problem
    res = solver(arg);

    // Optimal solution of the NLP
    V_opt = std::vector<double>(res.at("x"));
}

void extractSolution(
        const std::vector<double>& V_opt,
        const ProblemSize& ps,
        std::vector<s2mVector>& Q,
        std::vector<s2mVector>& Qdot,
        std::vector<s2mVector>& Tau)
{
    // Resizing the output variables
    for (unsigned int q=0; q<m.nbQ(); ++q){
        Tau.push_back(s2mTau(ps.ns));
        Q.push_back(s2mGenCoord(ps.ns+1));
        Qdot.push_back(s2mGenCoord(ps.ns+1));
    }

    // Get the optimal controls
    for(unsigned int i=0; i<ps.ns; ++i)
        for (unsigned int q=0; q<m.nbQ(); ++q)
            Tau[q][i] = V_opt.at(q + ps.nx + i*(ps.nx+m.nbQ()));

    // Get the states
    for(unsigned int i=0; i<ps.ns+1; ++i){
        for (unsigned int q=0; q<m.nbQ(); ++q){
            Q[q][i] = V_opt.at(q + i*(ps.nx+m.nbQ()));
            Qdot[q][i] = V_opt.at(q + m.nbQ() + i*(ps.nx+m.nbQ()));
        }
    }

}
