#ifndef VIOLIN_OPTIMIZATION_OBJECTIVES_H
#define VIOLIN_OPTIMIZATION_OBJECTIVES_H
#include "biorbd_declarer.h"
#include "dynamics.h"

void residualTorquesSquare( double *u, double *g, void *);
void muscleActivationsSquare( double *x, double *g, void *);
void stringToPlayObjective( double *x, double *g, void *user_data);
void accelerationsObjective( double *x, double *g, void *user_data);

#endif  // VIOLIN_OPTIMIZATION_OBJECTIVES_H
