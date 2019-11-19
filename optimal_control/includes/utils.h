#ifndef VIOLIN_OPTIMIZATION_UTILS_H
#define VIOLIN_OPTIMIZATION_UTILS_H
#include <iostream>
#include <errno.h>
#include <sys/stat.h>
#include "biorbd_declarer.h"

void createTreePath(const std::string& path);
bool dirExists(const char* const path);

void dispatchQ(const double *x);
void dispatchQandQdot(const double *x);
void dispatchActivation(const double *x);
void initializeMuscleStates();

void projectOnXyPlane(double *x, double *g, void *user_data);

void validityCheck();

#endif  // VIOLIN_OPTIMIZATION_UTILS_H
