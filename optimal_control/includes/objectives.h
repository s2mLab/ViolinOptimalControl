#ifndef __OBJECTIVES_H
#define __OBJECTIVES_H
#include "s2mMusculoSkeletalModel.h"
#include "dynamics.h"

void lagrangeResidualTorques( double *x, double *g, void *);
void lagrangeActivations( double *x, double *g, void *);
void lagrangeAccelerations( double *x, double *g, void *user_data);
void lagrangeTime( double *x, double *g, void *);
void mayerVelocity( double *x, double *g, void *);
void mayerRHS( double *x, double *g, void *user_data);

extern s2mMusculoSkeletalModel m;
extern unsigned int nQ;
extern unsigned int nQdot;
extern unsigned int nTau;
extern unsigned int nTags;
extern unsigned int nMus;
extern unsigned int nPhases;


#endif
