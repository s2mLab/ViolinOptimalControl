#ifndef __OBJECTIVES_H
#define __OBJECTIVES_H
#include "s2mMusculoSkeletalModel.h"
#include "dynamics.h"

void LagrangeResidualTorques( double *x, double *g, void *);
void LagrangeActivations( double *x, double *g, void *);
void LagrangeAccelerations( double *x, double *g, void *user_data);
void LagrangeTime( double *x, double *g, void *);
void MayerVelocity( double *x, double *g, void *);
void MayerRHS( double *x, double *g, void *user_data);

extern s2mMusculoSkeletalModel m;
extern unsigned int nQ;
extern unsigned int nQdot;
extern unsigned int nTau;
extern unsigned int nTags;
extern unsigned int nMus;


#endif
