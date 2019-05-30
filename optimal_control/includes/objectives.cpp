#include "objectives.h"


void LagrangeAddedTorques( double *x, double *g, void *){
    g[0]=0;
    for (unsigned int i=0; i<nTau; ++i)
        g[0]+=(x[i+nQ+nQdot+nMus]*x[i+nQ+nQdot+nMus]);
}

void LagrangeActivations( double *x, double *g, void *){
    g[0]=0;
    for (unsigned int i=0; i<nMus; ++i)
        g[0]+=(x[i+nQ+nQdot]*x[i+nQ+nQdot]);
}

void MayerSpeed( double *x, double *g, void *){
    g[0]=0;
    for (unsigned int i=0; i<nQdot; ++i)
        g[0]+=(x[i+nQ]*x[i+nQ]);
}

void MayerRHS( double *x, double *g, void *user_data){
    double * rhs = new double[nQ + nQdot];
    forwardDynamicsFromJointTorque(x, rhs, user_data);
    g[0] = 0;
    for (unsigned int i = 0; i<nQ+nQdot; ++i)
        g[0] += (rhs[i]*rhs[i]);
    delete[] rhs;
}


