#include "constraints.h"


void StatesZero( double *x, double *g, void * ){
    for (unsigned int i =0; i<nQ + nQdot; ++i) {
        g[i] =  x[i];
    }
}

void ActivationsZero( double *x, double *g, void * ){
    for (unsigned int i =0; i<nMus; ++i) {
        g[i] =  x[i+nQ+nQdot];
    }
}

void TorquesZero( double *x, double *g, void * ){
    for (unsigned int i =0; i<nTau; ++i) {
        g[i] =  x[i+nQ+nQdot+nMus];
    }
}

void Rotbras( double *x, double *g, void * ){
    g[0] = x[nQ-1]-PI/4;
}



