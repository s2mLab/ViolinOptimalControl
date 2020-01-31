#ifndef UTILS_CASADI_H
#define UTILS_CASADI_H

#include <casadi.hpp>
#include "biorbd.h"
extern biorbd::Model m;

#include <QtWidgets/QApplication>
#include <QtWidgets/QMainWindow>

struct ProblemSize{
    unsigned int ns; // number of shooting
    double tf; // Final time of the optimization
    double dt; // Time between two shooting

    unsigned int nu; // Number of controls
    unsigned int nx; // Number of states
};

struct BoundaryConditions{
    BoundaryConditions(
            std::vector<double> min = {},
            std::vector<double> max = {},
            std::vector<double> starting_min = {},
            std::vector<double> starting_max = {},
            std::vector<double> end_min = {},
            std::vector<double> end_max = {}
            ){
        // Dispatch intermediate states
        this->min = min;
        this->max = max;

        // If some are not declared put intermediate values instead
        starting_min.size() == 0
                ? this->starting_min = min
                : this->starting_min = starting_min;
        starting_max.size() == 0
                ? this->starting_max = max
                : this->starting_max = starting_max;
        end_min.size() == 0
                ? this->end_min = min
                : this->end_min = end_min;
        end_max.size() == 0
                ? this->end_max = max
                : this->end_max = end_max;
    }

    std::vector<double> min;
    std::vector<double> max;
    std::vector<double> starting_min;
    std::vector<double> starting_max;
    std::vector<double> end_min;
    std::vector<double> end_max;
};

struct InitialConditions{
    InitialConditions(
            std::vector<double> initialGuess = {}):
            val(initialGuess)
    {}
    std::vector<double> val;
};

enum Instant{
    START = 0,
    MID,
    END,
    ALL,
    NONE
};

struct IndexPairing{
    IndexPairing():
        t(Instant::NONE),
        toPair(std::vector<unsigned int>())
    {}
    IndexPairing(const Instant& _t,
                 std::vector<unsigned int> _idx):
        t(_t),
        toPair(_idx)
    {}
    unsigned int idx(unsigned i) const{
        return toPair[i];
    }
    Instant t;
    std::vector<unsigned int> toPair;
};

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
            Visualization::LEVEL level_,
            int argc, char *argv[]){
        level = level_;
        if (level > Visualization::LEVEL::NONE){
            app = new QApplication(argc, argv);
            window = new QMainWindow();
        }
    }
    QApplication *app;
    QMainWindow * window;
    LEVEL level;
};

enum PLANE{
    XY,
    YZ,
    XZ
};

enum ViolinStringNames{
    E,
    A,
    D,
    G,
    NO_STRING
};

enum ODE_SOLVER{
    COLLOCATION,
    RK,
    CVODES
};

void defineDifferentialVariables(
        ProblemSize &ps,
        casadi::MX &u,
        casadi::MX &x);

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
        std::vector<casadi::MX> &X);

///
/// This functions constraints the euler angle between a segment's
/// joint coordinate systeme (JCS) to a system of axes made from two
/// points and an origin to be 0.
/// segmentsToAlign is
/// 1) The index of the segment to align
/// 2) The index of the marker that describes the identity of the first axis (X, Y, Z)
/// 3) The index of the marker that describes the beginning of the first axis
/// 4) The index of the marker that describes the ending of the first axis
/// 5) The index of the marker that describes the identity of the second axis (X, Y, Z)
/// 6) The index of the marker that describes the beginning of the second axis
/// 7) The index of the marker that describes the ending of the second axis
/// 8) Which axis (X, Y, Z) to recalculate to
/// ensures orthonormal system of axis
///
void alignJcsToMarkersConstraint(
        const casadi::Function &dynamics,
        const casadi::Function &axesFunction,
        const ProblemSize &ps,
        const std::vector<casadi::MX> &U,
        const std::vector<casadi::MX> &X,
        std::vector<casadi::MX> &g,
        const std::vector<IndexPairing> &segmentsToAlign);

///
/// This functions constraints the dot product of a segment's axis and a
/// vector made from two points to be 1.
/// segmentsToAlign is
/// 1) The index of the segment to align
/// 2) The axis of the segment
/// 3) The index of the marker that describes the beginning of the vector to align
/// 4) The index of the marker that describes the ending of the vector to align
///
void alignAxesToMarkersConstraint(
        const casadi::Function &dynamics,
        const casadi::Function &axesFunction,
        const ProblemSize &ps,
        const std::vector<casadi::MX> &U,
        const std::vector<casadi::MX> &X,
        std::vector<casadi::MX> &g,
        const std::vector<IndexPairing> &segmentsToAlign);

///
/// This function constraints the dot product of two segments' axis to be 0.
/// segmentsToAlign is
/// 1) The index of the segment 1
/// 2) The axis of the segment 1
/// 3) the index of the segment 2
/// 3) the axis of the segment 2
///
void alignAxesConstraint(
        const casadi::Function &dynamics,
        const casadi::Function &axesFunction,
        const ProblemSize &ps,
        const std::vector<casadi::MX> &U,
        const std::vector<casadi::MX> &X,
        std::vector<casadi::MX> &g,
        const std::vector<IndexPairing> &segmentsToAlign);

///
/// This function constraints the projection of a point in the reference frame of
/// segment to be 0 on two axes (to lie on a plane).
/// projectionPolicy is
/// 1) Index of the segment to project on
/// 2) the index of the marker
/// 3) the plane to project on
///
void projectionOnPlaneConstraint(
        const casadi::Function &dynamics,
        const casadi::Function &forwardKin,
        const ProblemSize &ps,
        const std::vector<casadi::MX> &U,
        const std::vector<casadi::MX> &X,
        std::vector<casadi::MX> &g,
        const std::vector<IndexPairing> &projectionPolicy
        );

///
/// This function constraints the difference of the position of two markers to be 0.
/// markerIdx is
/// 1) The index of the marker 1
/// 2) The index of the marker 2
///
void followMarkerConstraint(
        const casadi::Function& dynamics,
        const casadi::Function &forwardKin,
        const ProblemSize& ps,
        const std::vector<casadi::MX> &U,
        const std::vector<casadi::MX> &X,
        std::vector<casadi::MX> &g,
        const std::vector<IndexPairing> &markerIdx);

///
/// This function ensures the end of a node to be equal to the next node
///
void continuityConstraints(
        const casadi::Function& dynamics,
        const ProblemSize& ps,
        const std::vector<casadi::MX> &U,
        const std::vector<casadi::MX> &X,
        std::vector<casadi::MX> &g);

void minimizeControls(
        const ProblemSize& ps,
        const std::vector<casadi::MX> &X,
        const std::vector<casadi::MX> &U,
        casadi::MX &obj);

///
/// \brief solveProblemWithIpopt Actually solving the problem
/// \param V The casadi State and Control variables
/// \param vBounds The boundaries for V
/// \param vInit The initial guesses for V
/// \param obj The objective function
/// \param constraints The constraint set
/// \param probSize The problem size
/// \param V_opt The optimized values (output)
/// \param animationLevel The level of online animation (0=None, 1=Charts, 2=Model visualization)
///
void solveProblemWithIpopt(
        const casadi::MX &V,
        const BoundaryConditions &vBounds,
        const InitialConditions &vInit,
        const casadi::MX &obj,
        const std::vector<casadi::MX> &constraints,
        const ProblemSize& probSize,
        std::vector<double>& V_opt,
        Visualization& visu);

void extractSolution(const std::vector<double>& V_opt,
        const ProblemSize& ps,
        std::vector<biorbd::utils::Vector> &Q,
        std::vector<biorbd::utils::Vector> &Qdot,
        std::vector<biorbd::utils::Vector> &u);

void createTreePath(const std::string& path);
bool dirExists(const char* const path);

template<class T>
void writeCasadiResults(
        const biorbd::utils::Path& path,
        const T& data,
        double dt){

    std::ofstream file;
    file.open(path.relativePath().c_str());
    if (!file){
        biorbd::utils::Error::raise("File " + path.relativePath() + " could not be open");
    }

    double currentTime(0);
    for (int j=0; j<data[0].size(); ++j){
        file << "[\t" << currentTime << "\t";
        for (int i=0; i<data.size(); ++i){
            file << data[i](j) << "\t";
        }
        file << "]\n";
        currentTime += dt;
    }
}

#include <iostream>
#include <chrono>
#include <thread>
#include <functional>
#include <mutex>
#include <condition_variable>
class AnimationCallback : public casadi::Callback {
public:
    AnimationCallback(
            Visualization& visu,
            const casadi::MX &V,
            const std::vector<casadi::MX> &constraints,
            const ProblemSize& probSize
        );

    void refresh();

//    casadi_int get_n_in() override { return 6;}
//    casadi_int get_n_out() override { return 1;}
//    virtual std::string get_name_in(casadi_int i) override{
//        if (i == 0){
//            return "x";
//        } else if (i == 1){
//            return "f";
//        } else if (i == 2){
//            return "g";
//        } else if (i == 3){
//            return "lam_x";
//        } else if (i == 4){
//            return "lam_g";
//        } else if (i == 5){
//            return "lam_p";
//        } else {
//            return "";
//        }
//    }
//    virtual casadi::Sparsity get_sparsity_in(casadi_int i) override{
//        if (i == 0){
//            return casadi::Sparsity::dense(_sparsityX);
//        } else if (i == 1){
//            return casadi::Sparsity::dense(1);
//        } else if (i == 2){
//            return casadi::Sparsity::dense(_sparsityG);
//        } else if (i == 3){
//            return casadi::Sparsity::dense(_sparsityX);
//        } else if (i == 4){
//            return casadi::Sparsity::dense(_sparsityG);
//        } else if (i == 5){
//            return casadi::Sparsity::dense(0);
//        } else {
//            return casadi::Sparsity::dense(-1);
//        }
//    }


//    virtual std::vector<casadi::DM> eval(const std::vector<casadi::DM>& arg) const override{
//        std::vector<biorbd::utils::Vector> Q;
//        std::vector<biorbd::utils::Vector> Qdot;
//        std::vector<biorbd::utils::Vector> Control;
//        extractSolution(std::vector<double>(arg[0]), _ps, Q, Qdot, Control);

//        for (unsigned int q=0; q<m.nbQ(); ++q){
//            for (int t=0; t<static_cast<int>(_ps.ns)+1; ++t){
//                _QSerie[q]->replace(t, _ps.dt*static_cast<double>(t), Q[q][t]);
//            }
//        }

//        for (unsigned int q=0; q<m.nbQdot(); ++q){
//            for (int t=0; t<static_cast<int>(_ps.ns)+1; ++t){
//                _QdotSerie[q]->replace(t, _ps.dt*static_cast<double>(t), Qdot[q][t]);
//            }
//        }

//        for (unsigned int q=0; q<m.nbGeneralizedTorque(); ++q){
//            for (int t=0; t<static_cast<int>(_ps.ns); ++t){
//                _TauSerie[q]->replace(t*2+0, _ps.dt*static_cast<double>(t), Control[q+m.nbMuscleTotal()][t]);
//                _TauSerie[q]->replace(t*2+1, _ps.dt*static_cast<double>(t), Control[q+m.nbMuscleTotal()][t]);
//            }
//        }
//        for (unsigned int q=0; q<m.nbMuscleTotal(); ++q){
//            for (int t=0; t<static_cast<int>(_ps.ns); ++t){
//                _MuscleSerie[q]->replace(t*2+0, _ps.dt*static_cast<double>(t), Control[q][t]);
//                _MuscleSerie[q]->replace(t*2+1, _ps.dt*static_cast<double>(t), Control[q][t]);
//            }
//        }


//        return {0};
//    }

protected:
    void wait_then_call(){
        std::unique_lock<std::mutex> lck{mtx};
        for (int i{10}; i>0; --i){
            std::cout << "coucou " << wait_thread.get_id() << " = " << i << std::endl;
            cv.wait_for(lck, time);
        }
    }

    QTimer * m_timer;
//    unsigned int _sparsityX;
//    unsigned int _sparsityG;
//    ProblemSize _ps;
    Visualization& _visu;

    std::mutex mtx;
    std::condition_variable cv{};
    std::chrono::milliseconds time;
    std::function <void(void)> f;
    std::thread wait_thread{[this]() {wait_then_call(); }};
//    std::vector<QtCharts::QLineSeries*> _QSerie;
//    std::vector<QtCharts::QLineSeries*> _QdotSerie;
//    std::vector<QtCharts::QLineSeries*> _TauSerie;
//    std::vector<QtCharts::QLineSeries*> _MuscleSerie;
};






#endif
