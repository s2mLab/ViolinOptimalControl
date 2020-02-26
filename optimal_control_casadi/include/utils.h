#ifndef UTILS_CASADI_H
#define UTILS_CASADI_H

#include <eigen3/Eigen/Dense>
#include <casadi.hpp>
#include "biorbdCasadi_interface_common.h"
#include "biorbd.h"
extern biorbd::Model m;
class AnimationCallback;

#include <QtWidgets/QApplication>
#include <QtWidgets/QMainWindow>
namespace QtCharts {
    class QLineSeries;
}

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
    INTERMEDIATES,
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

biorbd::utils::Vector ForwardDyn(
        biorbd::Model& model,
        const casadi::MX& states,
        const casadi::MX& controls);

void defineDifferentialVariables(
        ProblemSize &ps,
        casadi::MX &u,
        casadi::MX &x);


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
        bool useCyclicObjective,
        bool useCyclicConstraint,
        std::vector<void (*)(const ProblemSize&,
                             const std::vector<casadi::MX>&,
                             const std::vector<casadi::MX>&,
                             casadi::MX&)> objectiveFunctions,
        casadi::MX& V,
        BoundaryConditions& vBounds,
        InitialConditions& vInit,
        std::vector<casadi::MX>& g,
        BoundaryConditions& gBounds,
        casadi::MX& J);


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
        std::vector<casadi::MX> &X,
        bool useCyclicObjective = false);

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
void alignJcsToMarkersConstraint(const casadi::Function &dynamics,
        const ProblemSize &ps,
        const std::vector<casadi::MX> &U,
        const std::vector<casadi::MX> &X,
        const std::vector<IndexPairing> &segmentsToAlign,
        std::vector<casadi::MX> &g,
        BoundaryConditions &gBounds);

///
/// This functions constraints the dot product of a segment's axis and a
/// vector made from two points to be 1.
/// segmentsToAlign is
/// 1) The index of the segment to align
/// 2) The axis of the segment
/// 3) The index of the marker that describes the beginning of the vector to align
/// 4) The index of the marker that describes the ending of the vector to align
///
void alignAxesToMarkersConstraint(const casadi::Function &dynamics,
        const ProblemSize &ps,
        const std::vector<casadi::MX> &U,
        const std::vector<casadi::MX> &X,
        const std::vector<IndexPairing> &segmentsToAlign,
        std::vector<casadi::MX> &g,
        BoundaryConditions &gBounds);

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
        const ProblemSize &ps,
        const std::vector<casadi::MX> &U,
        const std::vector<casadi::MX> &X,
        const std::vector<IndexPairing> &segmentsToAlign,
        std::vector<casadi::MX> &g,
        BoundaryConditions& gBounds);

///
/// This function constraints the projection of a point in the reference frame of
/// segment to be 0 on two axes (to lie on a plane).
/// projectionPolicy is
/// 1) Index of the segment to project on
/// 2) the index of the marker
/// 3) the plane to project on
///
void projectionOnPlaneConstraint(const casadi::Function &dynamics,
        const ProblemSize &ps,
        const std::vector<casadi::MX> &U,
        const std::vector<casadi::MX> &X,
        const std::vector<IndexPairing> &projectionPolicy,
        std::vector<casadi::MX> &g,
        BoundaryConditions& gBounds);

///
/// This function constraints the difference of the position of two markers to be 0.
/// markerIdx is
/// 1) The index of the marker 1
/// 2) The index of the marker 2
///
void followMarkerConstraint(
        const casadi::Function& dynamics,
        const ProblemSize& ps,
        const std::vector<casadi::MX> &U,
        const std::vector<casadi::MX> &X,
        const std::vector<IndexPairing> &markerIdx,
        std::vector<casadi::MX> &g,
        BoundaryConditions& gBounds);

///
/// This function ensures the end of a node to be equal to the next node
///
void continuityConstraints(const casadi::Function& dynamics,
        const ProblemSize& ps,
        const std::vector<casadi::MX> &U,
        const std::vector<casadi::MX> &X,
        std::vector<casadi::MX> &g,
        BoundaryConditions& gBounds,
        bool isCyclic = false);

///
/// \brief cyclicObjective Objective that fit the last node with the first
///
void cyclicObjective(
        const ProblemSize &ps,
        const std::vector<casadi::MX> &X,
        const std::vector<casadi::MX> &U,
        std::vector<casadi::MX> &g,
        BoundaryConditions& gBounds,
        casadi::MX &obj);

///
/// \brief regulateStates Regulate function that minimizes all the states/1000
///
void regulateStates(
        const ProblemSize& ps,
        const std::vector<casadi::MX> &X,
        const std::vector<casadi::MX> &U,
        casadi::MX &obj);

///
/// \brief minimizeControls Objective function that minimizes all the controls
///
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
/// \param constraintsBounds The bondaries of the constraint set
/// \param probSize The problem size
/// \param animationLevel The level of online animation (0=None, 1=Charts, 2=Model visualization)
///
std::vector<double> solveProblemWithIpopt(const casadi::MX &V,
        const BoundaryConditions &vBounds,
        const InitialConditions &vInit,
        const casadi::MX &obj,
        const std::vector<casadi::MX> &constraints,
        const BoundaryConditions &constraintsBounds,
        const ProblemSize& probSize,
        AnimationCallback& visu);

void extractSolution(
        const std::vector<double>& V_opt,
        const ProblemSize& ps,
        std::vector<Eigen::VectorXd> &Q,
        std::vector<Eigen::VectorXd> &Qdot,
        std::vector<Eigen::VectorXd> &u);

void extractSolution(
        const std::vector<double>& V_opt,
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

///
/// \brief finalizeSolution Print the solution on screen and save to a file
///
void finalizeSolution(
        const std::vector<double>& V_opt,
        const ProblemSize& probSize,
        const std::string& optimizationName);

#endif
