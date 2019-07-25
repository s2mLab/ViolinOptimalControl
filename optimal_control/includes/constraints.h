#ifndef __CONSTRAINTS_H
#define __CONSTRAINTS_H
#include "s2mMusculoSkeletalModel.h"
#include "dynamics.h"

void statesZero( double *x, double *g, void * );
void velocityZero( double *x, double *g, void * );
void activationsZero( double *x, double *g, void * );
void torquesZero( double *x, double *g, void * );
void rotbras( double *x, double *g, void * );
void violonUp( double *x, double *g, void * );
void violonDown( double *x, double *g, void * );
void markerPosition(double *x, double *g, void *user_data );
void forceConstraintFromMuscleActivation( double *x, double *g, void *user_data);
void forceConstraintFromTorque(double *x, double *g, void *user_data);

extern s2mMusculoSkeletalModel m;
extern unsigned int nQ;
extern unsigned int nQdot;
extern unsigned int nTau;
extern unsigned int nTags;
extern unsigned int nMus;



#endif
