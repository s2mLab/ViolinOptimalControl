#ifndef __OBJECTIVES_H
#define __OBJECTIVES_H
#include "BiorbdModel.h"
#include "dynamics.h"

void lagrangeResidualTorques( double *u, double *g, void *);
void lagrangeResidualTorquesMultistage(double *u, double *g, void *);
void lagrangeActivations( double *x, double *g, void *);
void lagrangeAccelerations( double *x, double *g, void *user_data);
void lagrangeTime( double *x, double *g, void *);
void mayerVelocity( double *x, double *g, void *);
void mayerRHS( double *x, double *g, void *user_data);

extern biorbd::Model m;
extern unsigned int nQ;
extern unsigned int nQdot;
extern unsigned int nTau;
extern unsigned int nMarkers;
extern unsigned int nMus;
extern unsigned int nPhases;


#endif
