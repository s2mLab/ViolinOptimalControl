#include "utils.h"
#include <sys/stat.h>

#include "AnimationCallback.h"

casadi::MX sandbox(
        const casadi::MX& states,
        const IndexPairing& alignPolicy){
//    // Usage
//    casadi::MX qsym = casadi::MX::sym("q", m.nbQ(), 1);
//    casadi::Function f(casadi::Function(
//                           "sandbox",
//                           {qsym},
//                           {sandbox(qsym, alignPolicy)}));
//    std::cout << f(casadi::DMDict{{"i0", casadi::DM::zeros(m.nbQ(), 1)}}) << std::endl;

    // Get States
    biorbd::rigidbody::GeneralizedCoordinates q = states(
                casadi::Slice(0, static_cast<casadi_int>(m.nbQ())));


    // Get the system of axes of the segment to align
    unsigned int segmentIdx(alignPolicy.idx(0));
    biorbd::utils::Rotation r_seg(
                m.globalJCS(q, segmentIdx).rot());
    return r_seg;
}

// Biorbd interface
biorbd::utils::Vector ForwardDyn(
        const casadi::MX& states,
        const casadi::MX& controls)
{
    biorbd::rigidbody::GeneralizedCoordinates Q;
    biorbd::rigidbody::GeneralizedVelocity QDot;
    biorbd::rigidbody::GeneralizedAcceleration QDDot(m.nbQ());
    biorbd::rigidbody::GeneralizedTorque Tau(m.nbQ());
    unsigned int nMus(m.nbMuscleTotal());
    std::vector<std::shared_ptr<biorbd::muscles::StateDynamics>> musclesStates(nMus);
    for(unsigned int i = 0; i<nMus; ++i){
        musclesStates[i] = std::make_shared<biorbd::muscles::StateDynamics>(
                    biorbd::muscles::StateDynamics());
    }

    // Get States
    Q = states(casadi::Slice(0, static_cast<casadi_int>(m.nbQ())));
    QDot = states(casadi::Slice(static_cast<casadi_int>(m.nbQ()),
                                static_cast<casadi_int>(m.nbQ()*2)));

    // Get Controls
    if (nMus > 0){
        for (unsigned int i = 0; i<nMus; ++i) {
            musclesStates[i]->setActivation(controls(i, 0));
        }
        Tau = m.muscularJointTorque(musclesStates, true, &Q, &QDot);
    }
    else {
        Tau.setZero();
    }
    Tau += controls(casadi::Slice(
                        static_cast<casadi_int>(nMus),
                        static_cast<casadi_int>(nMus + m.nbQ())), 0);

    // Perform Forward Dynamics
    RigidBodyDynamics::ForwardDynamics(m, Q, QDot, Tau, QDDot);
    return vertcat(QDot, QDDot);
}

casadi::MX rungeKutta4(
        const casadi::Function &f,
        const ProblemSize& ps,
        const casadi::MX &U,
        const casadi::MX &X,
        unsigned int nStep){
    casadi::MX out(casadi::MX::zeros(ps.nx, 1));

    for (unsigned int i=0; i<nStep; ++i){ // loop over control intervals
        casadi::MX k1 = f(casadi::MXDict{{"states", X               }, {"controls", U}}).at("statesdot");
        casadi::MX k2 = f(casadi::MXDict{{"states", X + ps.dt/2 * k1}, {"controls", U}}).at("statesdot");
        casadi::MX k3 = f(casadi::MXDict{{"states", X + ps.dt/2 * k2}, {"controls", U}}).at("statesdot");
        casadi::MX k4 = f(casadi::MXDict{{"states", X + ps.dt   * k3}, {"controls", U}}).at("statesdot");
        out += ps.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
    }
    return out;
}

void prepareMusculoSkeletalNLP(
        ProblemSize& probSize,
        ODE_SOLVER odeSolver,
        const BoundaryConditions& uBounds,
        const InitialConditions& uInit,
        const BoundaryConditions& xBounds,
        const InitialConditions& xInit,
        const std::vector<IndexPairing> &markersToPair,
        const std::vector<IndexPairing> &markerToProject,
        const std::vector<IndexPairing> &axesToAlign,
        const std::vector<IndexPairing> &alignWithMarkers,
        const std::vector<IndexPairing> &alignWithMarkersReferenceFrame,
        const std::vector<IndexPairing> &alignWithCustomRT,
        bool useCyclicObjective,
        bool useCyclicConstraint,
        std::vector<std::pair<void (*)(const ProblemSize&,
                             const std::vector<casadi::MX>&,
                             const std::vector<casadi::MX>&,
                             double,
                             casadi::MX&), double>> objectiveFunctions,
        casadi::MX& V,
        BoundaryConditions& vBounds,
        InitialConditions& vInit,
        std::vector<casadi::MX>& g,
        BoundaryConditions& gBounds,
        casadi::MX& J,
        casadi::Function& dynamics
        ){
    // Differential variables
    casadi::MX u;
    casadi::MX x;
    defineDifferentialVariables(probSize, u, x);

    // Prepare the NLP problem
    std::vector<casadi::MX> U;
    std::vector<casadi::MX> X;
    defineMultipleShootingNodes(probSize, uBounds, xBounds, uInit, xInit,
                                V, vBounds, vInit, U, X);

    // ODE right hand side
    casadi::MX states = casadi::MX::sym("x", m.nbQ()*2, 1);
    casadi::MX controls = casadi::MX::sym("p", m.nbMuscleTotal() + m.nbQ(), 1);
    dynamics = casadi::Function( "ForwardDyn",
                                {states, controls},
                                {ForwardDyn(states, controls)},
                                {"states", "controls"},
                                {"statesdot"}).expand(); //.map(probSize.ns, "thread", 10);

    casadi::MXDict ode = {
        {"x", x},
        {"p", u},
        {"ode", dynamics(std::vector<casadi::MX>({x, u}))[0]}
    };
    casadi::Dict ode_opt;
    ode_opt["t0"] = 0;
    ode_opt["tf"] = probSize.dt;
    if (odeSolver == ODE_SOLVER::RK || odeSolver == ODE_SOLVER::COLLOCATION)
        ode_opt["number_of_finite_elements"] = 5;
    casadi::Function F;
    if (odeSolver == ODE_SOLVER::RK){
        F = casadi::integrator("integrator", "rk", ode, ode_opt);
//        F = casadi::Function("RK4", {states, controls}, {rungeKutta4(f, probSize, controls, states, 5)}, {"x0", "p"}, {"xf"});
    }
    else if (odeSolver == ODE_SOLVER::COLLOCATION)
        F = casadi::integrator("integrator", "collocation", ode, ode_opt);
    else if (odeSolver == ODE_SOLVER::CVODES)
        F = casadi::integrator("integrator", "cvodes", ode, ode_opt);
    else
        throw std::runtime_error("ODE solver not implemented..");

    // Continuity constraints
    continuityConstraints(F, probSize, U, X, g, gBounds, useCyclicConstraint);

    // Path constraints
    followMarkerConstraint(F, probSize, U, X, markersToPair, g, gBounds);

    // Path constraints
    projectionOnPlaneConstraint(F, probSize, U, X, markerToProject, g, gBounds);

    // Path constraints
    alignAxesConstraint(F, probSize, U, X, axesToAlign, g, gBounds);

    // Path constraints
    alignAxesToMarkersConstraint(F, probSize, U, X, alignWithMarkers, g, gBounds);

    // Path constraints
    alignJcsToMarkersConstraint(F, probSize, U, X, alignWithMarkersReferenceFrame, g, gBounds);

    // Path constraints
    alignWithCustomRTConstraint(F, probSize, U, X, alignWithCustomRT, g, gBounds);

    // Objective functions
    J = 0;
    for (unsigned int i=0; i<objectiveFunctions.size(); ++i){
        objectiveFunctions[i].first(probSize, X, U, objectiveFunctions[i].second, J);
    }

    if (useCyclicObjective){
        cyclicObjective(probSize, X, J);
    }
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
    casadi_assert(offset == static_cast<int>(NV), "");

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

void alignWithCustomRTConstraint(
        const casadi::Function &dynamics,
        const ProblemSize &ps,
        const std::vector<casadi::MX> &U,
        const std::vector<casadi::MX> &X,
        const std::vector<IndexPairing> &alignWithImu,
        std::vector<casadi::MX> &g,
        BoundaryConditions &gBounds){
    // Compute the state at final in case one pairing needs it
    casadi::MXDict I_end = dynamics(
                casadi::MXDict{{"x0", X[ps.ns-1]}, {"p", U[ps.ns-1]}});

    for (unsigned int p=0; p<alignWithImu.size(); ++p){
        const IndexPairing& alignPolicy(alignWithImu[p]);
        for (unsigned int t=0; t<ps.ns+1; ++t){
            casadi::MX x;
            if (!getState(t, ps, X, I_end, alignPolicy, x)){
                continue;
            }

            // Get the states
            const casadi::MX& q = x(casadi::Slice(0, static_cast<casadi_int>(m.nbQ())), 0);

            // Get the system of axes of the segment to align
            unsigned int segmentIdx(alignPolicy.idx(0));
            biorbd::utils::Rotation r_seg(
                        m.globalJCS(q, segmentIdx).rot());

            // Get the system of axes of the imu
            unsigned int rtIdx(alignPolicy.idx(1));
            biorbd::utils::Rotation r_rt(
                        m.RT(q, rtIdx).rot());

            casadi::MX angles = biorbd::utils::Rotation::toEulerAngles(r_seg.transpose() * r_rt, "zyx");
            g.push_back( angles );
            for (unsigned int i=0; i<angles.rows(); ++i){
                gBounds.min.push_back(0);
                gBounds.max.push_back(0);
            }
        }
    }
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

            // Get the states
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

            biorbd::utils::Matrix r_markers(
                        biorbd::utils::Rotation::fromMarkers(
                            {axis1Beg, axis1End}, {axis2Beg, axis2End}, {axis1name, axis2name},
                            axisToRecalculate));

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
            g.push_back( 1.0 - casadi::MX::dot(axes[0], axes[1]) );
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
            g.push_back( 1.0 - casadi::MX::dot(axes[0], axes[1]) );
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
        BoundaryConditions& gBounds,
        bool isCyclic)
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

    if (isCyclic){
        // Save continuity constraints between final integration and first node
        g.push_back( X[ps.ns] - X[0] );
        for (unsigned int i=0; i<m.nbQ()*2; ++i){
            gBounds.min.push_back(0);
            gBounds.max.push_back(0);
        }
    }
}

void cyclicObjective(
        const ProblemSize &ps,
        const std::vector<casadi::MX> &X,
        casadi::MX &obj){
    obj += casadi::MX::dot(X[0] - X[ps.ns], X[0] - X[ps.ns])*1000;
}

void minimizeStates(
        const ProblemSize &ps,
        const std::vector<casadi::MX> &X,
        const std::vector<casadi::MX> &,
        double weight,
        casadi::MX &obj)
{
    for(unsigned int k=0; k<ps.ns+1; ++k)
        obj += casadi::MX::dot(X[k], X[k])*ps.dt * weight;
}

void minimizeMuscleControls(
        const ProblemSize &ps,
        const std::vector<casadi::MX> &,
        const std::vector<casadi::MX> &U,
        double weight,
        casadi::MX &obj)
{
    for(unsigned int k=0; k<ps.ns; ++k)
        obj += casadi::MX::dot(
                    U[k](casadi::Slice(static_cast<casadi_int>(0),
                                       static_cast<casadi_int>(m.nbMuscleTotal())), 0),
                    U[k](casadi::Slice(static_cast<casadi_int>(0),
                                       static_cast<casadi_int>(m.nbMuscleTotal())), 0))*ps.dt * weight;
}

void minimizeTorqueControls(
        const ProblemSize &ps,
        const std::vector<casadi::MX> &,
        const std::vector<casadi::MX> &U,
        double weight,
        casadi::MX &obj)
{
    for(unsigned int k=0; k<ps.ns; ++k)
        obj += casadi::MX::dot(
                    U[k](casadi::Slice(static_cast<casadi_int>(m.nbMuscleTotal()),
                                       static_cast<casadi_int>(m.nbMuscleTotal() + m.nbQ())), 0),
                    U[k](casadi::Slice(static_cast<casadi_int>(m.nbMuscleTotal()),
                                       static_cast<casadi_int>(m.nbMuscleTotal() + m.nbQ())), 0))*ps.dt*weight;
}

void minimizeAllControls(
        const ProblemSize &ps,
        const std::vector<casadi::MX> &,
        const std::vector<casadi::MX> &U,
        int weight,
        casadi::MX &obj)
{
    for(unsigned int k=0; k<ps.ns; ++k)
        obj += casadi::MX::dot(U[k], U[k])*ps.dt*weight;
}

std::vector<double> solveProblemWithIpopt(
        const casadi::MX &V,
        const BoundaryConditions& vBounds,
        const InitialConditions& vInit,
        const casadi::MX &obj,
        const std::vector<casadi::MX> &constraints,
        const BoundaryConditions& constraintsBounds,
        const ProblemSize&,
        AnimationCallback &visuCallback)
{
    std::cout << "Solving the optimal control problem..." << std::endl;

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
    opts["ipopt.hessian_approximation"] = "exact"; // "exact", "limited-memory"
    opts["ipopt.limited_memory_max_history"] = 50;
    opts["ipopt.linear_solver"] = "ma57"; // "ma57", "ma86", "mumps"

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
    return std::vector<double>(res.at("x"));
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

void finalizeSolution(
        const std::vector<double>& V_opt,
        const ProblemSize& probSize,
        const std::string& optimizationName){
    std::vector<Eigen::VectorXd> Q;
    std::vector<Eigen::VectorXd> Qdot;
    std::vector<Eigen::VectorXd> Controls;
    extractSolution(V_opt, probSize, Q, Qdot, Controls);

//    // Show the solution
//    std::cout << "Results:" << std::endl;
//    for (unsigned int q=0; q<m.nbQ(); ++q){
//        std::cout << "Q[" << q <<"] = " << Q[q].transpose() << std::endl;
//        std::cout << "Qdot[" << q <<"] = " << Qdot[q].transpose() << std::endl;
//        std::cout << "Tau[" << q <<"] = " << Controls[q+m.nbMuscleTotal()].transpose() << std::endl;
//        std::cout << std::endl;
//    }
//    for (unsigned int q=0; q<m.nbMuscleTotal(); ++q){
//        std::cout << "Muscle[" << q <<"] = " << Controls[q].transpose() << std::endl;
//        std::cout << std::endl;
//    }

    const std::string resultsPath("../../Results/");
    const biorbd::utils::Path controlResultsFileName(resultsPath + "Controls" + optimizationName + ".txt");
    const biorbd::utils::Path stateResultsFileName(resultsPath + "States" + optimizationName + ".txt");

    createTreePath(resultsPath);
    std::vector<Eigen::VectorXd> QandQdot;
    for (auto q : Q){
        QandQdot.push_back(q);
    }
    for (auto qdot : Qdot){
        QandQdot.push_back(qdot);
    }
    writeCasadiResults(controlResultsFileName, Controls, probSize.dt);
    writeCasadiResults(stateResultsFileName, QandQdot, probSize.dt);
}
