#include "utils.h"
#include <sys/stat.h>

void defineDifferentialVariables(
        ProblemSize& ps,
        casadi::MX& u,
        casadi::MX& x)
{
    // Controls
    for (unsigned int i=0; i<m.nbGeneralizedTorque(); ++i)
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
        const BoundaryConditions &uBounds,
        const BoundaryConditions &xBounds,
        const InitialConditions &uInit,
        const InitialConditions &xInit,
        casadi::MX &V,
        BoundaryConditions &vBounds,
        InitialConditions &vInit,
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
            vBounds.min.insert(vBounds.min.end(), xBounds.starting_min.begin(), xBounds.starting_min.end());
            vBounds.max.insert(vBounds.max.end(), xBounds.starting_max.begin(), xBounds.starting_max.end());
        } else {
            vBounds.min.insert(vBounds.min.end(), xBounds.min.begin(), xBounds.min.end());
            vBounds.max.insert(vBounds.max.end(), xBounds.max.begin(), xBounds.max.end());
        }
        vInit.val.insert(vInit.val.end(), xInit.val.begin(), xInit.val.end());
        offset += ps.nx;

        // Local control
        U.push_back( V.nz(casadi::Slice(offset,offset+static_cast<int>(ps.nu))));
        vBounds.min.insert(vBounds.min.end(), uBounds.min.begin(), uBounds.min.end());
        vBounds.max.insert(vBounds.max.end(), uBounds.max.begin(), uBounds.max.end());
        vInit.val.insert(vInit.val.end(), uInit.val.begin(), uInit.val.end());
        offset += ps.nu;
    }

    // State at end
    X.push_back(V.nz(casadi::Slice(offset,offset+static_cast<int>(ps.nx))));
    vBounds.min.insert(vBounds.min.end(), xBounds.end_min.begin(), xBounds.end_min.end());
    vBounds.max.insert(vBounds.max.end(), xBounds.end_max.begin(), xBounds.end_max.end());
    vInit.val.insert(vInit.val.end(), xInit.val.begin(), xInit.val.end());
    offset += ps.nx;

    // Make sure that the size of the variable vector is consistent with the
    // number of variables that we have referenced
    casadi_assert(offset==static_cast<int>(NV), "");
}

void alignAxesToMarkersConstraint(
        const casadi::Function &dynamics,
        const casadi::Function &axesFunction,
        const ProblemSize &ps,
        const std::vector<casadi::MX> &U,
        const std::vector<casadi::MX> &X,
        std::vector<casadi::MX> &g,
        const std::vector<IndexPairing> &segmentsToAlign)
{
    // Compute the state at final in case one pairing needs it
    casadi::MXDict I_end = dynamics(
                casadi::MXDict{{"x0", X[ps.ns-1]}, {"p", U[ps.ns-1]}});

    for (unsigned int p=0; p<segmentsToAlign.size(); ++p){
        const IndexPairing& alignPolicy(segmentsToAlign[p]);
        for (unsigned int t=0; t<ps.ns+1; ++t){
            casadi::MX x;
            if (t == 0 && (alignPolicy.t == Instant::START || alignPolicy.t == Instant::ALL)){
                // If at starting point
                x = X[t];
            }
            else if (t == ps.ns && (alignPolicy.t == Instant::END || alignPolicy.t == Instant::ALL)){
                // If at end point
                x = I_end.at("xf");
            }
            else if (alignPolicy.t == Instant::MID || alignPolicy.t == Instant::ALL){
                // If at mid points
                x = X[t];
            }
            else {
                continue;
            }
            casadi::MX segmentIndexAndAxis(2, 1);
            segmentIndexAndAxis(0) = alignPolicy.idx(0);
            segmentIndexAndAxis(1) = alignPolicy.idx(1);
            casadi::MX markersIdx(2, 1);
            markersIdx(0) = alignPolicy.idx(2);
            markersIdx(1) = alignPolicy.idx(3);
            casadi::MXDict angle = axesFunction(casadi::MXDict{
                             {"States", x},
                             {"UpdateKinematics", true},
                             {"SegmentIndexAndAxis", segmentIndexAndAxis},
                             {"MarkersIndex", markersIdx}
                            });
            g.push_back( 1 - angle.at("Angle") );
        }
    }
}

void alignAxesConstraint(
        const casadi::Function &dynamics,
        const casadi::Function &axesFunction,
        const ProblemSize &ps,
        const std::vector<casadi::MX> &U,
        const std::vector<casadi::MX> &X,
        std::vector<casadi::MX> &g,
        const std::vector<IndexPairing> &segmentsToAlign)
{
    // Compute the state at final in case one pairing needs it
    casadi::MXDict I_end = dynamics(
                casadi::MXDict{{"x0", X[ps.ns-1]}, {"p", U[ps.ns-1]}});

    for (unsigned int p=0; p<segmentsToAlign.size(); ++p){
        const IndexPairing& alignPolicy(segmentsToAlign[p]);
        for (unsigned int t=0; t<ps.ns+1; ++t){
            casadi::MX x;
            if (t == 0 && (alignPolicy.t == Instant::START || alignPolicy.t == Instant::ALL)){
                // If at starting point
                x = X[t];
            }
            else if (t == ps.ns && (alignPolicy.t == Instant::END || alignPolicy.t == Instant::ALL)){
                // If at end point
                x = I_end.at("xf");
            }
            else if (alignPolicy.t == Instant::MID || alignPolicy.t == Instant::ALL){
                // If at mid points
                x = X[t];
            }
            else {
                continue;
            }
            casadi::MX firstSegmentIndexAndAxis(2, 1);
            firstSegmentIndexAndAxis(0) = alignPolicy.idx(0);
            firstSegmentIndexAndAxis(1) = alignPolicy.idx(1);
            casadi::MX secondSegmentIndexAndAxis(2, 1);
            secondSegmentIndexAndAxis(0) = alignPolicy.idx(2);
            secondSegmentIndexAndAxis(1) = alignPolicy.idx(3);
            casadi::MXDict angle = axesFunction(casadi::MXDict{
                             {"States", x},
                             {"UpdateKinematics", true},
                             {"FirstSegmentIndexAndAxis", firstSegmentIndexAndAxis},
                             {"SecondSegmentIndexAndAxis", secondSegmentIndexAndAxis}
                            });
            g.push_back( 1 - angle.at("Angle") );
        }
    }
}

void projectionOnPlaneConstraint(
        const casadi::Function &dynamics,
        const casadi::Function &projectionFunction,
        const ProblemSize &ps,
        const std::vector<casadi::MX> &U,
        const std::vector<casadi::MX> &X,
        std::vector<casadi::MX> &g,
        const std::vector<IndexPairing> &projectionPolicy
        )
{
    // Compute the state at final in case one pairing needs it
    casadi::MXDict I_end = dynamics(
                casadi::MXDict{{"x0", X[ps.ns-1]}, {"p", U[ps.ns-1]}});

    for (auto policy : projectionPolicy){
        for (unsigned int t=0; t<ps.ns+1; ++t){
            casadi::MX x;
            if (t == 0 && (policy.t == Instant::START || policy.t == Instant::ALL)){
                // If at starting point
                x = X[t];
            }
            else if (t == ps.ns && (policy.t == Instant::END || policy.t == Instant::ALL)){
                // If at end point
                x = I_end.at("xf");
            }
            else if (policy.t == Instant::MID || policy.t == Instant::ALL){
                // If at mid points
                x = X[t];
            }
            else {
                continue;
            }
            casadi::MXDict M(projectionFunction(casadi::MXDict{
                    {"States", x},
                    {"UpdateKinematics", true},
                    {"SegmentToProjectOnIndex", policy.idx(0)},
                    {"MarkerToProjectIndex", policy.idx(1)},
                    }));

            if (policy.idx(2) == PLANE::XY){
                g.push_back( M.at("ProjectedMarker")(0, 0) );
                g.push_back( M.at("ProjectedMarker")(1, 0) );
            }
            else if (policy.idx(2) == PLANE::YZ){
                g.push_back( M.at("ProjectedMarker")(1, 0) );
                g.push_back( M.at("ProjectedMarker")(2, 0) );
            }
            else if (policy.idx(2) == PLANE::XZ){
                g.push_back( M.at("ProjectedMarker")(0, 0) );
                g.push_back( M.at("ProjectedMarker")(2, 0) );
            }

        }
    }
}

void followMarkerConstraint(
        const casadi::Function &dynamics,
        const casadi::Function &forwardKin,
        const ProblemSize &ps,
        const std::vector<casadi::MX> &U,
        const std::vector<casadi::MX> &X,
        std::vector<casadi::MX> &g,
        const std::vector<IndexPairing>& markerIdx
        )
{
    // Compute the state at final in case one pairing needs it
    casadi::MXDict I_end = dynamics(
                casadi::MXDict{{"x0", X[ps.ns-1]}, {"p", U[ps.ns-1]}});

    for (auto pair : markerIdx){
        for (unsigned int t=0; t<ps.ns+1; ++t){
            casadi::MX x;
            if (t == 0 && (pair.t == Instant::START || pair.t == Instant::ALL)){
                // If at starting point
                x = X[t];
            }
            else if (t == ps.ns && (pair.t == Instant::END || pair.t == Instant::ALL)){
                // If at end point
                x = I_end.at("xf");
            }
            else if (pair.t == Instant::MID || pair.t == Instant::ALL){
                // If at mid points
                x = X[t];
            }
            else {
                continue;
            }

            casadi::MXDict M1(forwardKin(casadi::MXDict{
                    {"States", x},
                    {"UpdateKinematics", true},
                    {"MarkerIndex", pair.idx(0)}
                    }));
            casadi::MXDict M2(forwardKin(casadi::MXDict{
                    {"States", x},
                    {"UpdateKinematics", false},
                    {"MarkerIndex", pair.idx(1)}
                    }));
            g.push_back( M1.at("Marker") - M2.at("Marker") );
        }
    }
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
        obj += casadi::MX::dot(U[k], U[k])*ps.dt;
}

void solveProblemWithIpopt(
        const casadi::MX &V,
        const BoundaryConditions& vBounds,
        const InitialConditions& vInit,
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
    opts["ipopt.tol"] = 1e-6;
    opts["ipopt.max_iter"] = 1000;
    opts["ipopt.hessian_approximation"] = "limited-memory";

    // Create an NLP solver and buffers
    casadi::Function solver = nlpsol("nlpsol", "ipopt", nlp, opts);
    std::map<std::string, casadi::DM> arg, res;

    // Bounds and initial guess
    arg["lbx"] = vBounds.min;
    arg["ubx"] = vBounds.max;
    arg["lbg"] = 0;
    arg["ubg"] = 0;
    arg["x0"] = vInit.val;

    // Solve the problem
    res = solver(arg);

    // Optimal solution of the NLP
    V_opt = std::vector<double>(res.at("x"));
}

void extractSolution(
        const std::vector<double>& V_opt,
        const ProblemSize& ps,
        std::vector<biorbd::rigidbody::GeneralizedCoordinates>& Q,
        std::vector<biorbd::rigidbody::GeneralizedVelocity>& Qdot,
        std::vector<biorbd::rigidbody::GeneralizedTorque>& Tau)
{
    // Resizing the output variables
    for (unsigned int q=0; q<m.nbQ(); ++q){
        Tau.push_back(biorbd::rigidbody::GeneralizedTorque(ps.ns));
        Q.push_back(biorbd::rigidbody::GeneralizedCoordinates(ps.ns+1));
        Qdot.push_back(biorbd::rigidbody::GeneralizedVelocity(ps.ns+1));
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

void createTreePath(const std::string &path)
{
    if (!dirExists(path.c_str()))
        mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
}

bool dirExists(const char* const path)
{
    struct stat info;

    int statRC = stat( path, &info );
    if( statRC != 0 )
    {
        if (errno == ENOENT)  { return 0; } // something along the path does not exist
        if (errno == ENOTDIR) { return 0; } // something in path prefix is not a dir
        return -1;
    }

    return ( info.st_mode & S_IFDIR ) ? true : false;
}
