#ifndef ANIMATION_CALLBACK_H
#define ANIMATION_CALLBACK_H

#include <iostream>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <casadi.hpp>
#include "biorbd.h"
#include "utils.h"
extern biorbd::Model m;

#include <QtWidgets/QApplication>
#include <QtWidgets/QMainWindow>
namespace QtCharts {
    class QLineSeries;
}

struct Visualization{
    enum LEVEL{
        NONE,
        GRAPH,
        THREE_DIMENSION
    };
    Visualization(){
        level = Visualization::LEVEL::NONE;
    }
    Visualization(
            Visualization::LEVEL level,
            int argc, char *argv[]):
        argc(argc),
        argv(argv),
        level(level){
    }
    int argc;
    char **argv;
    LEVEL level;
};

class AnimationCallback : public casadi::Callback {
public:
    AnimationCallback(
            Visualization& visu,
            const casadi::MX &V,
            const std::vector<casadi::MX> &constraints,
            const ProblemSize& probSize,
            size_t time,
            const casadi::Function& dynamics);

    casadi_int get_n_in() override;
    casadi_int get_n_out() override;
    virtual std::string get_name_in(casadi_int i) override;
    virtual casadi::Sparsity get_sparsity_in(casadi_int i) override;


    virtual std::vector<casadi::DM> eval(const std::vector<casadi::DM>& arg) const override;

    bool isActive() const;
    Visualization visu() const;

protected:
    void _rungeKutta4(
            casadi::DM Qinit,
            casadi::DM QDotinit,
            casadi::DM Tau,
            const casadi::DM& muscleActivations,
            std::vector<casadi::DM>& QOut,
            std::vector<casadi::DM>& QDotOut) const;
    const casadi::Function& _dynamicsFunc;
    mutable bool _isReady;

    void QtWindowThread();
    unsigned int _sparsityX;
    unsigned int _sparsityG;
    ProblemSize _ps;
    Visualization& _visu;
    std::vector<QtCharts::QLineSeries*> _QSerie;
    std::vector<QtCharts::QLineSeries*> _QdotSerie;
    std::vector<QtCharts::QLineSeries*> _TauSerie;
    std::vector<QtCharts::QLineSeries*> _MuscleSerie;

    QApplication * _app;
    QMainWindow * _window;
    std::chrono::milliseconds refreshTime;
    std::thread * refresh_thread;
    unsigned int _nbElementsRK4;
    bool _isActive = true;
};

#endif // ANIMATION_CALLBACK_H
