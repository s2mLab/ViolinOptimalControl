#ifndef VIOLIN_OPTIMIZATION_CONSTRAINTS_H
#define VIOLIN_OPTIMIZATION_CONSTRAINTS_H
#include "biorbd_declarer.h"
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
void orthogonalProjected(double *x, double *g, void *user_data);



#endif  // VIOLIN_OPTIMIZATION_CONSTRAINTS_H
