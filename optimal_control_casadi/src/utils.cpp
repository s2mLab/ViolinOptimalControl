#include "utils.h"
#include <sys/stat.h>

#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QVBoxLayout>

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


void alignJcsToMarkersConstraint(
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
            casadi::MX axis1(3, 1);
            axis1(0) = alignPolicy.idx(1);
            axis1(1) = alignPolicy.idx(2);
            axis1(2) = alignPolicy.idx(3);
            casadi::MX axis2(3, 1);
            axis2(0) = alignPolicy.idx(4);
            axis2(1) = alignPolicy.idx(5);
            axis2(2) = alignPolicy.idx(6);
            casadi::MXDict angle = axesFunction(casadi::MXDict{
                             {"States", x},
                             {"UpdateKinematics", true},
                             {"SegmentIndex", alignPolicy.idx(0)},
                             {"Axis1Description", axis1},
                             {"Axis2Description", axis2},
                             {"AxisToRecalculate", alignPolicy.idx(7)},
                            });
            g.push_back( angle.at("Angles") );
        }
    }
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

class AnimationCallback : public casadi::Callback {
public:
    AnimationCallback(
            Visualization& visu,
            const casadi::MX &V,
            const std::vector<casadi::MX> &constraints,
            const ProblemSize& probSize) :
        _visu(visu){

        _ps = probSize;

        _sparsityX = V.size().first;
        _sparsityG = 0;
        for (int i=0; i<constraints.size(); ++i){
            _sparsityG += constraints[i].size().first;
        }

        // Qrange
        std::vector<biorbd::utils::Range> ranges;
        for (unsigned int i=0; i<m.nbSegment(); ++i){
            std::vector<biorbd::utils::Range> segRanges(m.segment(i).ranges());
            for(unsigned int j=0; j<segRanges.size(); ++j){
                ranges.push_back(segRanges[j]);
            }
        }

        // Create the Qt visualistion
        QWidget * mainWidget = new QWidget();
        _visu.window->setCentralWidget(mainWidget);
        _visu.window->resize(1000, 500);
        _visu.window->show();

        QVBoxLayout * qLayout = new QVBoxLayout();
        for (unsigned int i=0; i<m.nbQ(); ++i){
            _QSerie.push_back(new QtCharts::QLineSeries());
            for (unsigned int j=0; j<probSize.ns+1; ++j){
                _QSerie[i]->append(0, 0);
            }
            QtCharts::QChart *chart = new QtCharts::QChart();
            chart->legend()->hide();
            chart->addSeries(_QSerie[i]);
            chart->createDefaultAxes();
            chart->setTitle((m.nameDof()[i] + ", Q").c_str());
            chart->axes(Qt::Horizontal)[0]->setMin(0);
            chart->axes(Qt::Horizontal)[0]->setMax(static_cast<double>(probSize.ns) * probSize.dt);
            chart->axes(Qt::Vertical)[0]->setMin(ranges[i].min());
            chart->axes(Qt::Vertical)[0]->setMax(ranges[i].max());
            chart->setMinimumHeight(200);
            QtCharts::QChartView * chartView = new QtCharts::QChartView(chart);
            chartView->setRenderHint(QPainter::Antialiasing);
            qLayout->addWidget(chartView);
        }
        QVBoxLayout * qdotLayout = new QVBoxLayout();
        for (unsigned int i=0; i<m.nbQdot(); ++i){
            _QdotSerie.push_back(new QtCharts::QLineSeries());
            for (unsigned int j=0; j<probSize.ns+1; ++j){
                _QdotSerie[i]->append(0, 0);
            }
            QtCharts::QChart *chart = new QtCharts::QChart();
            chart->legend()->hide();
            chart->addSeries(_QdotSerie[i]);
            chart->createDefaultAxes();
            chart->setTitle((m.nameDof()[i] + ", Qdot").c_str());
            chart->axes(Qt::Horizontal)[0]->setMin(0);
            chart->axes(Qt::Horizontal)[0]->setMax(static_cast<double>(probSize.ns) * probSize.dt);
            chart->axes(Qt::Vertical)[0]->setMin(-10);
            chart->axes(Qt::Vertical)[0]->setMax(10);
            chart->setMinimumHeight(200);
            QtCharts::QChartView *chartView = new QtCharts::QChartView(chart);
            chartView->setRenderHint(QPainter::Antialiasing);
            qdotLayout->addWidget(chartView);
        }
        QVBoxLayout * tauLayout = new QVBoxLayout();
        for (unsigned int i=0; i<m.nbGeneralizedTorque(); ++i){
            _TauSerie.push_back(new QtCharts::QLineSeries());
            for (unsigned int j=0; j<probSize.ns; ++j){
                _TauSerie[i]->append(0, 0);
            }
            QtCharts::QChart *chart = new QtCharts::QChart();
            chart->legend()->hide();
            chart->addSeries(_TauSerie[i]);
            chart->createDefaultAxes();
            chart->setTitle((m.nameDof()[i] + ", Tau").c_str());
            chart->axes(Qt::Horizontal)[0]->setMin(0);
            chart->axes(Qt::Horizontal)[0]->setMax(static_cast<double>(probSize.ns) * probSize.dt);
            chart->axes(Qt::Vertical)[0]->setMin(-10);
            chart->axes(Qt::Vertical)[0]->setMax(10);
            chart->setMinimumHeight(200);
            QtCharts::QChartView *chartView = new QtCharts::QChartView(chart);
            chartView->setRenderHint(QPainter::Antialiasing);
            tauLayout->addWidget(chartView);
        }
        QVBoxLayout * muscleLayout = new QVBoxLayout();
        for (unsigned int i=0; i<m.nbMuscleTotal(); ++i){
            _MuscleSerie.push_back(new QtCharts::QLineSeries());
            QtCharts::QChart *chart = new QtCharts::QChart();
            chart->legend()->hide();
            chart->addSeries(_MuscleSerie[i]);
            chart->createDefaultAxes();
            chart->setTitle((m.muscleNames()[i]).c_str());
            chart->axes(Qt::Horizontal)[0]->setMin(0);
            chart->axes(Qt::Horizontal)[0]->setMax(static_cast<double>(probSize.ns) * probSize.dt);
            chart->axes(Qt::Vertical)[0]->setMin(0);
            chart->axes(Qt::Vertical)[0]->setMax(1);
            chart->setMinimumHeight(200);
            QtCharts::QChartView *chartView = new QtCharts::QChartView(chart);
            chartView->setRenderHint(QPainter::Antialiasing);
            muscleLayout->addWidget(chartView);
        }
        QHBoxLayout * allLayout = new QHBoxLayout();

        QWidget * qWidget = new QWidget();
        qWidget->setLayout(qLayout);
        QScrollArea * scrollQArea = new QScrollArea();
        scrollQArea->setFrameShape(QFrame::Shape::StyledPanel);
        scrollQArea->setWidgetResizable(true);
        scrollQArea->setWidget(qWidget);
        allLayout->addWidget(scrollQArea);

        QWidget * qdotWidget = new QWidget();
        qdotWidget->setLayout(qdotLayout);
        QScrollArea * scrollQDotArea = new QScrollArea();
        scrollQDotArea->setFrameShape(QFrame::Shape::StyledPanel);
        scrollQDotArea->setWidgetResizable(true);
        scrollQDotArea->setWidget(qdotWidget);
        allLayout->addWidget(scrollQDotArea);

        QWidget * tauWidget = new QWidget();
        tauWidget->setLayout(tauLayout);
        QScrollArea * scrollTauArea = new QScrollArea();
        scrollTauArea->setFrameShape(QFrame::Shape::StyledPanel);
        scrollTauArea->setWidgetResizable(true);
        scrollTauArea->setWidget(tauWidget);
        allLayout->addWidget(scrollTauArea);

        mainWidget->setLayout(allLayout);

        construct("Callback");
    }

    casadi_int get_n_in() override { return 6;}
    casadi_int get_n_out() override { return 1;}
    virtual std::string get_name_in(casadi_int i) override{
        if (i == 0){
            return "x";
        } else if (i == 1){
            return "f";
        } else if (i == 2){
            return "g";
        } else if (i == 3){
            return "lam_x";
        } else if (i == 4){
            return "lam_g";
        } else if (i == 5){
            return "lam_p";
        } else {
            return "";
        }
    }
    virtual casadi::Sparsity get_sparsity_in(casadi_int i) override{
        if (i == 0){
            return casadi::Sparsity::dense(_sparsityX);
        } else if (i == 1){
            return casadi::Sparsity::dense(1);
        } else if (i == 2){
            return casadi::Sparsity::dense(_sparsityG);
        } else if (i == 3){
            return casadi::Sparsity::dense(_sparsityX);
        } else if (i == 4){
            return casadi::Sparsity::dense(_sparsityG);
        } else if (i == 5){
            return casadi::Sparsity::dense(0);
        } else {
            return casadi::Sparsity::dense(-1);
        }
    }


    virtual std::vector<casadi::DM> eval(const std::vector<casadi::DM>& arg) const override{
        std::vector<biorbd::utils::Vector> Q;
        std::vector<biorbd::utils::Vector> Qdot;
        std::vector<biorbd::utils::Vector> Control;
        extractSolution(std::vector<double>(arg[0]), _ps, Q, Qdot, Control);

        for (unsigned int q=0; q<m.nbQ(); ++q){
            for (unsigned int t=0; t<_ps.ns+1; ++t){
                _QSerie[q]->replace(t, _ps.dt*static_cast<double>(t), Q[q][t]);
            }
        }

        for (unsigned int q=0; q<m.nbQdot(); ++q){
            for (unsigned int t=0; t<_ps.ns+1; ++t){
                _QdotSerie[q]->replace(t, _ps.dt*static_cast<double>(t), Qdot[q][t]);
            }
        }

        for (unsigned int q=0; q<m.nbGeneralizedTorque(); ++q){
            for (unsigned int t=0; t<_ps.ns; ++t){
                _TauSerie[q]->replace(t, _ps.dt*static_cast<double>(t), Control[q+m.nbMuscleTotal()][t]);

            }
        }
        _visu.app->processEvents();

        return {0};
    }

protected:
    unsigned int _sparsityX;
    unsigned int _sparsityG;
    ProblemSize _ps;
    Visualization& _visu;

    std::vector<QtCharts::QLineSeries*> _QSerie;
    std::vector<QtCharts::QLineSeries*> _QdotSerie;
    std::vector<QtCharts::QLineSeries*> _TauSerie;
    std::vector<QtCharts::QLineSeries*> _MuscleSerie;
};

void solveProblemWithIpopt(
        const casadi::MX &V,
        const BoundaryConditions& vBounds,
        const InitialConditions& vInit,
        const casadi::MX &obj,
        const std::vector<casadi::MX> &constraints,
        const ProblemSize& probSize,
        std::vector<double>& V_opt,
        Visualization &visu)
{
    // NLP
    casadi::MXDict nlp = {{"x", V},
                          {"f", obj},
                          {"g", vertcat(constraints)}};

    // Set options
    casadi::Dict opts;
    AnimationCallback callback(visu, V, constraints, probSize);
    if (visu.level > Visualization::LEVEL::NONE){
        opts["iteration_callback"] = callback;
    }
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
        std::vector<biorbd::utils::Vector>& Q,
        std::vector<biorbd::utils::Vector>& Qdot,
        std::vector<biorbd::utils::Vector>& u)
{
    // Resizing the output variables
    for (unsigned int q=0; q<m.nbQ(); ++q){
        u.push_back(biorbd::utils::Vector(ps.ns));
        Q.push_back(biorbd::rigidbody::GeneralizedCoordinates(ps.ns+1));
        Qdot.push_back(biorbd::rigidbody::GeneralizedVelocity(ps.ns+1));
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
