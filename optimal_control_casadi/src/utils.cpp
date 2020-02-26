#include "utils.h"
#include <sys/stat.h>

#include "AnimationCallback.h"

// Biorbd interface
biorbd::utils::Vector ForwardDyn(
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

void defineDifferentialVariables(
        ProblemSize& ps,
        casadi::MX& u,
        casadi::MX& x)
{
    // Controls
    std::vector<biorbd::utils::String> muscleNames(m.muscleNames());
    std::vector<biorbd::utils::String> dofNames(m.nameDof());

    for (unsigned int i=0; i<m.nbMuscleTotal(); ++i){
        u = vertcat(u, casadi::MX::sym("Control_" + muscleNames[i]));
    }
    for (unsigned int i=0; i<m.nbGeneralizedTorque(); ++i){
        u = vertcat(u, casadi::MX::sym("Control_" + dofNames[i]));
    }

    // States
    casadi::MX q;
    casadi::MX qdot;
    for (unsigned int i=0; i<m.nbQ(); ++i){
        q = vertcat(q, casadi::MX::sym("Q_" + m.nameDof()[i]));
        qdot = vertcat(qdot, casadi::MX::sym("Qdot_" + dofNames[i]));
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


bool getState(
        unsigned int t,
        const ProblemSize &ps,
        const std::vector<casadi::MX> &X,
        const casadi::MXDict& I_end,
        const IndexPairing& alignPolicy,
        casadi::MX& x){
    if (alignPolicy.t == Instant::MID && ps.ns % 2 != 0){
        biorbd::utils::Error::raise("To use Instant::MID, you must have a pair number of shooting points");
    }

    if (t == 0 && (alignPolicy.t == Instant::START || alignPolicy.t == Instant::ALL)){
        // If at starting point
        x = X[t];
    }
    else if (t == ps.ns && (alignPolicy.t == Instant::END || alignPolicy.t == Instant::ALL)){
        // If at end point
        x = I_end.at("xf");
    }
    else if (t == ps.ns / 2 && alignPolicy.t == Instant::MID){
        x = X[t];
    }
    else if (alignPolicy.t == Instant::INTERMEDIATES || alignPolicy.t == Instant::ALL){
        // If at mid points
        x = X[t];
    }
    else {
        return false;
    }
    return true;
}

void alignJcsToMarkersConstraint(
        const casadi::Function &dynamics,
        const ProblemSize &ps,
        const std::vector<casadi::MX> &U,
        const std::vector<casadi::MX> &X,
        const std::vector<IndexPairing> &segmentsToAlign,
        std::vector<casadi::MX> &g,
        BoundaryConditions& gBounds)
{
    // Compute the state at final in case one pairing needs it
    casadi::MXDict I_end = dynamics(
                casadi::MXDict{{"x0", X[ps.ns-1]}, {"p", U[ps.ns-1]}});

    for (unsigned int p=0; p<segmentsToAlign.size(); ++p){
        const IndexPairing& alignPolicy(segmentsToAlign[p]);
        for (unsigned int t=0; t<ps.ns+1; ++t){
            casadi::MX x;
            if (!getState(t, ps, X, I_end, alignPolicy, x)){
                continue;
            }

            // Get the angle between the two reference frames
            const casadi::MX& q = x(casadi::Slice(0, static_cast<casadi_int>(m.nbQ())), 0);

            // Get the system of axes of the segment to align
            unsigned int segmentIdx(alignPolicy.idx(0));
            biorbd::utils::Rotation r_seg(
                        m.globalJCS(q, segmentIdx).rot());

            // Get the system of axes from the markers
            biorbd::utils::String axis1name(
                        getAxisInString(static_cast<AXIS>(alignPolicy.idx(1))));
            biorbd::rigidbody::NodeSegment axis1Beg(
                        m.marker(q, alignPolicy.idx(2)));
            biorbd::rigidbody::NodeSegment axis1End(
                        m.marker(q, alignPolicy.idx(3)));

            biorbd::utils::String axis2name(
                        getAxisInString(static_cast<AXIS>(alignPolicy.idx(4))));
            biorbd::rigidbody::NodeSegment axis2Beg(
                        m.marker(q, alignPolicy.idx(5)));
            biorbd::rigidbody::NodeSegment axis2End(
                        m.marker(q, alignPolicy.idx(6)));

            biorbd::utils::String axisToRecalculate(
                        getAxisInString(static_cast<AXIS>(alignPolicy.idx(7))));

            biorbd::utils::Rotation r_markers(
                        biorbd::utils::Rotation::fromMarkers(
                            {axis1Beg, axis1End}, {axis2Beg, axis2End}, {axis1name, axis2name},
                            axisToRecalculate));

            // Get the angle between the two reference frames
            casadi::MX angles = biorbd::utils::Rotation::toEulerAngles(r_seg.transpose() * r_markers, "zyx");
            g.push_back( angles );
            for (unsigned int i=0; i<angles.rows(); ++i){
                gBounds.min.push_back(0);
                gBounds.max.push_back(0);
            }

        }
    }
}

void alignAxesToMarkersConstraint(
        const casadi::Function &dynamics,
        const ProblemSize &ps,
        const std::vector<casadi::MX> &U,
        const std::vector<casadi::MX> &X,
        const std::vector<IndexPairing> &segmentsToAlign,
        std::vector<casadi::MX> &g,
        BoundaryConditions& gBounds)
{
    // Compute the state at final in case one pairing needs it
    casadi::MXDict I_end = dynamics(
                casadi::MXDict{{"x0", X[ps.ns-1]}, {"p", U[ps.ns-1]}});

    for (unsigned int p=0; p<segmentsToAlign.size(); ++p){
        const IndexPairing& alignPolicy(segmentsToAlign[p]);
        for (unsigned int t=0; t<ps.ns+1; ++t){
            casadi::MX x;
            if (!getState(t, ps, X, I_end, alignPolicy, x)){
                continue;
            }
            const casadi::MX& q = x(casadi::Slice(0, static_cast<casadi_int>(m.nbQ())), 0);

            // Get the RT for the segment
            unsigned int segmentIdx = alignPolicy.idx(0);
            biorbd::utils::Rotation rt(m.globalJCS(q, segmentIdx).rot());

            // Extract the respective axes
            std::vector<biorbd::utils::Vector> axes;
            AXIS axisIdx = static_cast<AXIS>(alignPolicy.idx(1));
            if (axisIdx == AXIS::X){
                axes.push_back(rt.axe(0));
            } else if (axisIdx == AXIS::MINUS_X){
                axes.push_back(-rt.axe(0));
            } else if (axisIdx == AXIS::Y){
                axes.push_back(rt.axe(1));
            } else if (axisIdx == AXIS::MINUS_Y){
                axes.push_back(-rt.axe(1));
            } else if (axisIdx == AXIS::Z){
                axes.push_back(rt.axe(2));
            } else if (axisIdx == AXIS::MINUS_Z){
                axes.push_back(-rt.axe(2));
            }

            // Get the second axis by subtracting the two markers
            unsigned int markersIdx1(alignPolicy.idx(2));
            unsigned int markersIdx2(alignPolicy.idx(3));
            axes.push_back(
                        m.marker(q, markersIdx2)
                        - m.marker(q, markersIdx1));

            // Return the answers
            g.push_back( 1.0 - axes[0].dot(axes[1]) );
            gBounds.min.push_back(0);
            gBounds.max.push_back(0);
        }
    }
}

void alignAxesConstraint(
        const casadi::Function &dynamics,
        const ProblemSize &ps,
        const std::vector<casadi::MX> &U,
        const std::vector<casadi::MX> &X,
        const std::vector<IndexPairing> &segmentsToAlign,
        std::vector<casadi::MX> &g,
        BoundaryConditions& gBounds)
{
    // Compute the state at final in case one pairing needs it
    casadi::MXDict I_end = dynamics(
                casadi::MXDict{{"x0", X[ps.ns-1]}, {"p", U[ps.ns-1]}});

    for (unsigned int p=0; p<segmentsToAlign.size(); ++p){
        const IndexPairing& alignPolicy(segmentsToAlign[p]);
        for (unsigned int t=0; t<ps.ns+1; ++t){
            casadi::MX x;
            if (!getState(t, ps, X, I_end, alignPolicy, x)){
                continue;
            }
            const casadi::MX& q = x(casadi::Slice(0, static_cast<casadi_int>(m.nbQ())), 0);

            std::vector<biorbd::utils::Vector> axes;
            for (unsigned int i=0; i<2; ++i){
                // Get the RT for the segment
                unsigned int segmentRtIdx = alignPolicy.idx(i*2);
                biorbd::utils::Rotation rt(m.globalJCS(q, segmentRtIdx).rot());

                // Extract the respective axes
                AXIS axisIdx = static_cast<AXIS>(alignPolicy.idx(i*2+1));
                if (axisIdx == AXIS::X){
                    axes.push_back(rt.axe(0));
                } else if (axisIdx == AXIS::MINUS_X){
                    axes.push_back(-rt.axe(0));
                } else if (axisIdx == AXIS::Y){
                    axes.push_back(rt.axe(1));
                } else if (axisIdx == AXIS::MINUS_Y){
                    axes.push_back(-rt.axe(1));
                } else if (axisIdx == AXIS::Z){
                    axes.push_back(rt.axe(2));
                } else if (axisIdx == AXIS::MINUS_Z){
                    axes.push_back(-rt.axe(2));
                }
            }

            // The axes are align if they are colinear
            g.push_back( 1.0 - axes[0].dot(axes[1]) );
            gBounds.min.push_back(0);
            gBounds.max.push_back(0);
        }
    }
}


void projectionOnPlaneConstraint(
        const casadi::Function &dynamics,
        const ProblemSize &ps,
        const std::vector<casadi::MX> &U,
        const std::vector<casadi::MX> &X,
        const std::vector<IndexPairing> &projectionPolicy,
        std::vector<casadi::MX> &g,
        BoundaryConditions& gBounds
        )
{
    // Compute the state at final in case one pairing needs it
    casadi::MXDict I_end = dynamics(
                casadi::MXDict{{"x0", X[ps.ns-1]}, {"p", U[ps.ns-1]}});

    for (auto policy : projectionPolicy){
        for (unsigned int t=0; t<ps.ns+1; ++t){
            casadi::MX x;
            if (!getState(t, ps, X, I_end, policy, x)){
                continue;
            }
            const casadi::MX& q = x(casadi::Slice(0, static_cast<casadi_int>(m.nbQ())), 0);

            // Project marker on the RT of a specific segment
            biorbd::utils::RotoTrans rt(m.globalJCS(q, policy.idx(0)));
            biorbd::rigidbody::NodeSegment M(m.marker(q, policy.idx(1)));
            M.applyRT(rt.transpose());

            if (policy.idx(2) == PLANE::XY){
                g.push_back( M(0, 0) );
                g.push_back( M(1, 0) );
            }
            else if (policy.idx(2) == PLANE::YZ){
                g.push_back( M(1, 0) );
                g.push_back( M(2, 0) );
            }
            else if (policy.idx(2) == PLANE::XZ){
                g.push_back( M(0, 0) );
                g.push_back( M(2, 0) );
            }
            gBounds.min.push_back(0);
            gBounds.max.push_back(0);
            gBounds.min.push_back(0);
            gBounds.max.push_back(0);
        }
    }
}

void followMarkerConstraint(
        const casadi::Function &dynamics,
        const ProblemSize &ps,
        const std::vector<casadi::MX> &U,
        const std::vector<casadi::MX> &X,
        const std::vector<IndexPairing>& markerIdx,
        std::vector<casadi::MX> &g,
        BoundaryConditions& gBounds
        )
{
    // Compute the state at final in case one pairing needs it
    casadi::MXDict I_end = dynamics(
                casadi::MXDict{{"x0", X[ps.ns-1]}, {"p", U[ps.ns-1]}});

    for (auto pair : markerIdx){
        for (unsigned int t=0; t<ps.ns+1; ++t){
            casadi::MX x;
            if (!getState(t, ps, X, I_end, pair, x)){
                continue;
            }
            const casadi::MX& q = x(casadi::Slice(0, static_cast<casadi_int>(m.nbQ())), 0);

            casadi::MX M1_2 = m.marker(q, pair.idx(0));
            casadi::MX M2_2 = m.marker(q, pair.idx(1));
            g.push_back( casadi::MX::dot(M1_2 - M2_2, M1_2 - M2_2)  );
            gBounds.min.push_back(0);
            gBounds.max.push_back(0);
        }
    }
}

void continuityConstraints(
        const casadi::Function &dynamics,
        const ProblemSize &ps,
        const std::vector<casadi::MX> &U,
        const std::vector<casadi::MX> &X,
        std::vector<casadi::MX> &g,
        BoundaryConditions& gBounds)
{
    // Loop over shooting nodes
    for(unsigned int k=0; k<ps.ns; ++k){
        // Create an evaluation node
        casadi::MXDict I_out = dynamics(casadi::MXDict{{"x0", X[k]}, {"p", U[k]}});

        // Save continuity constraints
        g.push_back( I_out.at("xf") - X[k+1] );
        for (unsigned int i=0; i<m.nbQ()*2; ++i){
            gBounds.min.push_back(0);
            gBounds.max.push_back(0);
        }
    }
}

void cyclicConstraints(
        const casadi::Function &dynamics,
        const ProblemSize &ps,
        const std::vector<casadi::MX> &U,
        const std::vector<casadi::MX> &X,
        std::vector<casadi::MX> &g,
        BoundaryConditions& gBounds)
{
    // Create an evaluation node for the end point
    casadi::MXDict I_out = dynamics(casadi::MXDict{{"x0", X[ps.ns-1]}, {"p", U[ps.ns-1]}});

    // Save continuity constraints between final integration and first node
    g.push_back( I_out.at("xf") - X[0] );
    for (unsigned int i=0; i<m.nbQ()*2; ++i){
        gBounds.min.push_back(0);
        gBounds.max.push_back(0);
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
        const BoundaryConditions& constraintsBounds,
        const ProblemSize& probSize,
        std::vector<double>& V_opt,
        AnimationCallback &visuCallback)
{
    // NLP
    casadi::MXDict nlp = {{"x", V},
                          {"f", obj},
                          {"g", vertcat(constraints)}};

    // Set options
    casadi::Dict opts;
    if (visuCallback.visu().level > Visualization::LEVEL::NONE){
        opts["iteration_callback"] = visuCallback;
    }

    opts["ipopt.tol"] = 1e-6;
    opts["ipopt.max_iter"] = 1000;
//    opts["ipopt.hessian_approximation"] = "limited-memory";

    // Create an NLP solver and buffers
    casadi::Function solver = nlpsol("nlpsol", "ipopt", nlp, opts);
    std::map<std::string, casadi::DM> arg, res;

    // Bounds and initial guess
    arg["lbx"] = vBounds.min;
    arg["ubx"] = vBounds.max;
    arg["lbg"] = constraintsBounds.min;
    arg["ubg"] = constraintsBounds.max;
    arg["x0"] = vInit.val;

    // Solve the problem
    res = solver(arg);

    // Optimal solution of the NLP
    V_opt = std::vector<double>(res.at("x"));
}

void extractSolution(
        const std::vector<double>& V_opt,
        const ProblemSize& ps,
        std::vector<Eigen::VectorXd>& Q,
        std::vector<Eigen::VectorXd>& Qdot,
        std::vector<Eigen::VectorXd>& u)
{
    // Resizing the output variables
    for (unsigned int q=0; q<m.nbMuscleTotal(); ++q){
        u.push_back(Eigen::VectorXd(ps.ns));
    }
    for (unsigned int q=0; q<m.nbQ(); ++q){
        u.push_back(Eigen::VectorXd(ps.ns));
        Q.push_back(Eigen::VectorXd(ps.ns+1));
        Qdot.push_back(Eigen::VectorXd(ps.ns+1));
    }

    // Get the optimal controls
    for(unsigned int i=0; i<ps.ns; ++i)
        for (unsigned int q=0; q<ps.nu; ++q)
            u[q][i] = V_opt.at(q + ps.nx + i*(ps.nx+ps.nu));

    // Get the states
    for(unsigned int i=0; i<ps.ns+1; ++i){
        for (unsigned int q=0; q<m.nbQ(); ++q){
            Q[q][i] = V_opt.at(q + i*(ps.nx+ps.nu));
            Qdot[q][i] = V_opt.at(q + m.nbQ() + i*(ps.nx+ps.nu));
        }
    }
}

void extractSolution(
        const std::vector<double>& V_opt,
        const ProblemSize& ps,
        std::vector<biorbd::utils::Vector>& Q,
        std::vector<biorbd::utils::Vector>& Qdot,
        std::vector<biorbd::utils::Vector>& u)
{
    // Resizing the output variables
    for (unsigned int q=0; q<m.nbMuscleTotal(); ++q){
        u.push_back(biorbd::utils::Vector(ps.ns));
    }
    for (unsigned int q=0; q<m.nbQ(); ++q){
        u.push_back(biorbd::utils::Vector(ps.ns));
        Q.push_back(biorbd::utils::Vector(ps.ns+1));
        Qdot.push_back(biorbd::utils::Vector(ps.ns+1));
    }

    // Get the optimal controls
    for(unsigned int i=0; i<ps.ns; ++i)
        for (unsigned int q=0; q<ps.nu; ++q)
            u[q][i] = V_opt.at(q + ps.nx + i*(ps.nx+ps.nu));

    // Get the states
    for(unsigned int i=0; i<ps.ns+1; ++i){
        for (unsigned int q=0; q<m.nbQ(); ++q){
            Q[q][i] = V_opt.at(q + i*(ps.nx+ps.nu));
            Qdot[q][i] = V_opt.at(q + m.nbQ() + i*(ps.nx+ps.nu));
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

