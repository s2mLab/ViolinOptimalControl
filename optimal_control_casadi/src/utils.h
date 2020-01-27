#ifndef UTILS_CASADI_H
#define UTILS_CASADI_H

#include <casadi.hpp>
#include "biorbd.h"
extern biorbd::Model m;

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

void alignAxesToMarkersConstraint(
        const casadi::Function &dynamics,
        const casadi::Function &axesFunction,
        const ProblemSize &ps,
        const std::vector<casadi::MX> &U,
        const std::vector<casadi::MX> &X,
        std::vector<casadi::MX> &g,
        const std::vector<IndexPairing> &segmentsToAlign);

void alignAxesConstraint(
        const casadi::Function &dynamics,
        const casadi::Function &axesFunction,
        const ProblemSize &ps,
        const std::vector<casadi::MX> &U,
        const std::vector<casadi::MX> &X,
        std::vector<casadi::MX> &g,
        const std::vector<IndexPairing> &segmentsToAlign);

void projectionOnPlaneConstraint(
        const casadi::Function &dynamics,
        const casadi::Function &forwardKin,
        const ProblemSize &ps,
        const std::vector<casadi::MX> &U,
        const std::vector<casadi::MX> &X,
        std::vector<casadi::MX> &g,
        const std::vector<IndexPairing> &projectionPolicy
        );

void followMarkerConstraint(
        const casadi::Function& dynamics,
        const casadi::Function &forwardKin,
        const ProblemSize& ps,
        const std::vector<casadi::MX> &U,
        const std::vector<casadi::MX> &X,
        std::vector<casadi::MX> &g,
        const std::vector<IndexPairing> &markerIdx);

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

void solveProblemWithIpopt(
        const casadi::MX &V,
        const BoundaryConditions &vBounds,
        const InitialConditions &vInit,
        const casadi::MX &obj,
        const std::vector<casadi::MX> &constraints,
        std::vector<double>& V_opt);

void extractSolution(
        const std::vector<double>& V_opt,
        const ProblemSize& ps,
        std::vector<biorbd::rigidbody::GeneralizedCoordinates>& Q,
        std::vector<biorbd::rigidbody::GeneralizedVelocity>& Qdot,
        std::vector<biorbd::rigidbody::GeneralizedTorque>& Tau);

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
#endif
