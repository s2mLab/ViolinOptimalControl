#include "AnimationCallback.h"

#include <eigen3/Eigen/Dense>
#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QVBoxLayout>

AnimationCallback::AnimationCallback(
        Visualization &visu,
        const casadi::MX &V,
        const std::vector<casadi::MX> &constraints,
        const ProblemSize& probSize,
        size_t refreshTime,
        const casadi::Function &dynamics) :
    _dynamicsFunc(dynamics),
    _visu(visu),
    refreshTime(std::chrono::milliseconds{refreshTime})
{

    _ps = probSize;

    _sparsityX = static_cast<unsigned int>(V.size().first);
    _sparsityG = 0;
    for (unsigned int i=0; i<constraints.size(); ++i){
        _sparsityG += constraints[i].size().first;
    }

    construct("Callback");

    // Start and wait for the thread that will redraw the figure at a fixed rate to be ready
    refresh_thread = new std::thread{[this]() {QtWindowThread(); }};
    while (!_isReady){

    }
}

casadi_int AnimationCallback::get_n_in() { return 6;}

casadi_int AnimationCallback::get_n_out() { return 1;}

std::string AnimationCallback::get_name_in(casadi_int i){
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

casadi::Sparsity AnimationCallback::get_sparsity_in(casadi_int i){
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

std::vector<casadi::DM> AnimationCallback::eval(
        const std::vector<casadi::DM> &arg) const {

    if (!_isReady){
        return {0};
    }

    _isReady = false;
    if (_visu.level != Visualization::LEVEL::NONE && _window->isVisible()){
        std::vector<Eigen::VectorXd> Q;
        std::vector<Eigen::VectorXd> Qdot;
        std::vector<Eigen::VectorXd> Control;
        extractSolution(std::vector<double>(arg[0]), _ps, Q, Qdot, Control);

        for (int t=0; t<static_cast<int>(_ps.ns); ++t){
            casadi::DM QInit(m.nbQ(), 1);
            casadi::DM QDotInit(m.nbQ(), 1);
            for (unsigned int i=0; i<m.nbQ(); ++i){
                QInit(i) = Q[i][t];
                QDotInit(i) = Qdot[i][t];
            }
            casadi::DM muscleActivations(m.nbMuscleTotal(), 1);
            for (unsigned int i=0; i<m.nbMuscleTotal(); ++i){
                muscleActivations(i) = Control[i][t];
            }
            casadi::DM Tau(m.nbQ(), 1);
            for (unsigned int i=0; i<m.nbQ(); ++i){
                Tau(i) = Control[i][t];
            }

            std::vector<casadi::DM> QInt;
            std::vector<casadi::DM> QDotInt;
            _rungeKutta4(QInit, QDotInit,
                         muscleActivations, Tau,
                         QInt, QDotInt);

            for (int t2=0; t2<static_cast<int>(_nbElementsRK4+1); ++t2){
                for (unsigned int q=0; q<m.nbQ(); ++q){
                    _QSerie[q]->replace(t*_nbElementsRK4+t2,
                                        _ps.dt*static_cast<double>(t)+ _ps.dt/static_cast<double>(_nbElementsRK4)*static_cast<double>(t2),
                                        static_cast<double>(QInt[t2](q)));
                }

                for (unsigned int q=0; q<m.nbQ(); ++q){
                    _QdotSerie[q]->replace(t*_nbElementsRK4+t2,
                                        _ps.dt*static_cast<double>(t)+ _ps.dt/static_cast<double>(_nbElementsRK4)*static_cast<double>(t2),
                                        static_cast<double>(QDotInt[t2](q)));
                }
            }
        }

        for (unsigned int q=0; q<m.nbGeneralizedTorque(); ++q){
            for (int t=0; t<static_cast<int>(_ps.ns); ++t){
                _TauSerie[q]->replace(t*2+0, _ps.dt*static_cast<double>(t), Control[q+m.nbMuscleTotal()][t]);
                _TauSerie[q]->replace(t*2+1, _ps.dt*static_cast<double>(t+1), Control[q+m.nbMuscleTotal()][t]);
            }
        }
        for (unsigned int q=0; q<m.nbMuscleTotal(); ++q){
            for (int t=0; t<static_cast<int>(_ps.ns); ++t){
                _MuscleSerie[q]->replace(t*2+0, _ps.dt*static_cast<double>(t), Control[q][t]);
                _MuscleSerie[q]->replace(t*2+1, _ps.dt*static_cast<double>(t+1), Control[q][t]);
            }
        }
    }
    _isReady = true;
    return {0};
}

bool AnimationCallback::isActive() const
{
    return _isActive;
}

Visualization AnimationCallback::visu() const
{
    return _visu;
}

void AnimationCallback::_rungeKutta4(
        casadi::DM QInit,
        casadi::DM QDotInit,
        casadi::DM Tau,
        const casadi::DM& muscleActivations,
        std::vector<casadi::DM>& QOut,
        std::vector<casadi::DM>& QDotOut) const {

    QOut.clear();
    QDotOut.clear();
    QOut.push_back(QInit);
    QDotOut.push_back(QDotInit);

    ProblemSize ps(_ps);
    ps.dt = ps.dt/static_cast<double>(_nbElementsRK4);
    casadi::DM X(vertcat(QInit, QDotInit));
    casadi::DM U(vertcat(muscleActivations, Tau));
    for (unsigned int i=0; i<_nbElementsRK4; ++i){
        _app->processEvents();
        casadi::DM k1 = _dynamicsFunc(casadi::DMDict{{"states", X               }, {"controls", U}}).at("statesdot");
        casadi::DM k2 = _dynamicsFunc(casadi::DMDict{{"states", X + ps.dt/2 * k1}, {"controls", U}}).at("statesdot");
        casadi::DM k3 = _dynamicsFunc(casadi::DMDict{{"states", X + ps.dt/2 * k2}, {"controls", U}}).at("statesdot");
        casadi::DM k4 = _dynamicsFunc(casadi::DMDict{{"states", X + ps.dt   * k3}, {"controls", U}}).at("statesdot");
        X += ps.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
        QOut.push_back(X(casadi::Slice(0, static_cast<casadi_int>(m.nbQ())), 0));
        QDotOut.push_back(X(casadi::Slice(static_cast<casadi_int>(m.nbQ()), static_cast<casadi_int>(m.nbQ()*2)), 0));
    }
}

void AnimationCallback::QtWindowThread(){
    if (_visu.level == Visualization::LEVEL::NONE){
        _isActive = false;
        return;
    }

    _isReady = true;
    _nbElementsRK4 = 5;
    _isActive = true;
    _app = new QApplication(_visu.argc, _visu.argv);
    _window = new QMainWindow();

    // Create the Qt visualistion
    QWidget * mainWidget = new QWidget();
    _window->setCentralWidget(mainWidget);
    _window->resize(1000, 500);
    _window->show();

    // Qrange
    std::vector<biorbd::utils::Range> ranges;
    for (unsigned int i=0; i<m.nbSegment(); ++i){
        std::vector<biorbd::utils::Range> segRanges(m.segment(i).ranges());
        for(unsigned int j=0; j<segRanges.size(); ++j){
            ranges.push_back(segRanges[j]);
        }
    }

    QVBoxLayout * qLayout = new QVBoxLayout();
    for (unsigned int i=0; i<m.nbQ(); ++i){
        _QSerie.push_back(new QtCharts::QLineSeries());
        for (unsigned int j=0; j<(_ps.ns*_nbElementsRK4)+1; ++j){
            _QSerie[i]->append(0, 0);
        }
        QtCharts::QChart *chart = new QtCharts::QChart();
        chart->legend()->hide();
        chart->addSeries(_QSerie[i]);
        chart->createDefaultAxes();
        chart->setTitle((m.nameDof()[i] + ", Q").c_str());
        chart->axes(Qt::Horizontal)[0]->setMin(0);
        chart->axes(Qt::Horizontal)[0]->setMax(static_cast<double>(_ps.ns) * _ps.dt);
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
        for (unsigned int j=0; j<(_ps.ns*_nbElementsRK4)+1; ++j){
            _QdotSerie[i]->append(0, 0);
        }
        QtCharts::QChart *chart = new QtCharts::QChart();
        chart->legend()->hide();
        chart->addSeries(_QdotSerie[i]);
        chart->createDefaultAxes();
        chart->setTitle((m.nameDof()[i] + ", Qdot").c_str());
        chart->axes(Qt::Horizontal)[0]->setMin(0);
        chart->axes(Qt::Horizontal)[0]->setMax(static_cast<double>(_ps.ns) * _ps.dt);
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
        for (unsigned int j=0; j<_ps.ns; ++j){
            _TauSerie[i]->append(0, 0);
            _TauSerie[i]->append(0, 0);
        }
        QtCharts::QChart *chart = new QtCharts::QChart();
        chart->legend()->hide();
        chart->addSeries(_TauSerie[i]);
        chart->createDefaultAxes();
        chart->setTitle((m.nameDof()[i] + ", Tau").c_str());
        chart->axes(Qt::Horizontal)[0]->setMin(0);
        chart->axes(Qt::Horizontal)[0]->setMax(static_cast<double>(_ps.ns) * _ps.dt);
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
        for (unsigned int j=0; j<_ps.ns; ++j){
            _MuscleSerie[i]->append(0, 0);
            _MuscleSerie[i]->append(0, 0);
        }
        QtCharts::QChart *chart = new QtCharts::QChart();
        chart->legend()->hide();
        chart->addSeries(_MuscleSerie[i]);
        chart->createDefaultAxes();
        chart->setTitle((m.muscleNames()[i]).c_str());
        chart->axes(Qt::Horizontal)[0]->setMin(0);
        chart->axes(Qt::Horizontal)[0]->setMax(static_cast<double>(_ps.ns) * _ps.dt);
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

    if (m.nbMuscleTotal() > 0){
        QWidget * muscleWidget = new QWidget();
        muscleWidget->setLayout(muscleLayout);
        QScrollArea * scrollMuscleArea = new QScrollArea();
        scrollMuscleArea->setFrameShape(QFrame::Shape::StyledPanel);
        scrollMuscleArea->setWidgetResizable(true);
        scrollMuscleArea->setWidget(muscleWidget);
        allLayout->addWidget(scrollMuscleArea);
    }

    mainWidget->setLayout(allLayout);


    std::mutex mtx;
    std::unique_lock<std::mutex> lck{mtx};
    std::condition_variable cv{};
    _isReady = true;
    while (_window->isVisible()) {
        _window->repaint();
        _app->processEvents();
        cv.wait_for(lck, refreshTime);
    }
    _isActive = false;
}
