#ifndef __UTILS_H
#define __UTILS_H
#include <iostream>
#include <errno.h>
#include <sys/stat.h>
#include "biorbd/BiorbdModel.h"

using namespace biorbd::rigidbody;

void createTreePath(const std::string& path);
bool dirExists(const char* const path);
void Dispatch_Q_Qdot(double *x);
void Dispatch_Q_Qdot_Tau(double *x);

extern biorbd::Model m;
extern unsigned int nQ;
extern unsigned int nQdot;
extern unsigned int nTau;

extern GeneralizedCoordinates Q, Qdot, Qddot;
extern GeneralizedTorque Tau;


#endif
