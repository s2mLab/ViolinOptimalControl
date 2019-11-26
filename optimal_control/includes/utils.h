#ifndef VIOLIN_OPTIMIZATION_UTILS_H
#define VIOLIN_OPTIMIZATION_UTILS_H
#include <iostream>
#include <fstream>
#include <errno.h>
#include <sys/stat.h>
#include "biorbd_declarer.h"
#include <acado_optimal_control.hpp>

void createTreePath(const std::string& path);
bool dirExists(const char* const path);

void dispatchQ(const double *x);
void dispatchQandQdot(const double *x);
void dispatchActivation(const double *x);
void initializeMuscleStates();

void projectOnXyPlane(double *x, double *g, void *user_data);
void removeSquareBracketsInFile(
        const std::string& originFilePath,
        const std::string& targetFilePath);
ACADO::VariablesGrid readStates(
        const std::string& stateFilePath,
        const int nPoints,
        const int nPhases,
        const double t_Start,
        const double t_End);

ACADO::VariablesGrid readControls(
        const std::string& controlFilePath,
        const int nPoints,
        const int nPhases,
        const double t_Start,
        const double t_End);

void validityCheck();

#endif  // VIOLIN_OPTIMIZATION_UTILS_H
