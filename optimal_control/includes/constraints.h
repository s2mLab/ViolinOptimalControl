#ifndef __CONSTRAINTS_H
#define __CONSTRAINTS_H
#include "s2mMusculoSkeletalModel.h"

void StatesZero( double *x, double *g, void * );
void VelocityZero( double *x, double *g, void * );
void ActivationsZero( double *x, double *g, void * );
void TorquesZero( double *x, double *g, void * );
void Rotbras( double *x, double *g, void * );
void ViolonUp( double *x, double *g, void * );
void ViolonDown( double *x, double *g, void * );
void MarkerPosition(double *x, double *g, void *user_data );

extern s2mMusculoSkeletalModel m;
extern unsigned int nQ;
extern unsigned int nQdot;
extern unsigned int nTau;
extern unsigned int nTags;
extern unsigned int nMus;



#endif
