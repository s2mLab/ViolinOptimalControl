#ifndef __DYNAMICS_H
#define __DYNAMICS_H
#include "s2mMusculoSkeletalModel.h"

void LagrangeTorques( double *x, double *g, void *);
void MayerSpeed( double *x, double *g, void *);
void MayerRHS( double *x, double *g, void *);

extern s2mMusculoSkeletalModel m;
extern unsigned int nQ;
extern unsigned int nQdot;
extern unsigned int nTau;
extern unsigned int nTags;
extern unsigned int nMus;


#endif
