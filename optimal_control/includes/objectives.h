#ifndef VIOLIN_OPTIMIZATION_OBJECTIVES_H
#define VIOLIN_OPTIMIZATION_OBJECTIVES_H
#include "biorbd_declarer.h"
#include "dynamics.h"

void residualTorquesSquare( double *u, double *g, void *);
void lagrangeResidualTorquesMultistage(double *u, double *g, void *);
void bowDirectionAgainstViolin( double *x, double *g, void *user_data);
void muscleActivationsSquare( double *x, double *g, void *);
void lagrangeAccelerations( double *x, double *g, void *user_data);
void lagrangeTime( double *x, double *g, void *);
void mayerVelocity( double *x, double *g, void *);
void mayerRHS( double *x, double *g, void *user_data);

#endif  // VIOLIN_OPTIMIZATION_OBJECTIVES_H
