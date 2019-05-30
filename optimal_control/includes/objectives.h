#ifndef __OBJECTIVES_H
#define __OBJECTIVES_H
#include "s2mMusculoSkeletalModel.h"
#include "dynamics.h"

void LagrangeAddedTorques( double *x, double *g, void *);
void LagrangeActivations( double *x, double *g, void *);
void MayerSpeed( double *x, double *g, void *);
void MayerRHS( double *x, double *g, void *user_data);

extern s2mMusculoSkeletalModel m;
extern unsigned int nQ;
extern unsigned int nQdot;
extern unsigned int nTau;
extern unsigned int nTags;
extern unsigned int nMus;


#endif
