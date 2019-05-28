#include "constraints.h"

#define  NI   nQ + nQdot         // number of initial value constraints
void myInitialValueConstraint( double *x, double *g, void * ){
    for (unsigned int i =0; i<nQ + nQdot; ++i) {
        g[i] =  x[i]-0.1;
    }
}

#define  NE   1                 // number of end-point / terminal constraints
void myEndPointConstraint( double *x, double *g, void * ){
    g[0]=x[nQ-1]-PI/4;                         // rotation de 90°
//    for (unsigned int i=0; i<nQ-1; ++i)
//        g[i] = x[i]-0.01;
}





