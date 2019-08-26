#include "objectives.h"

static unsigned int i;

void lagrangeResidualTorques( double *x, double *g, void *){
    g[0]=0;
    for (i=0; i<nTau; ++i)
        g[0]+=(x[i+nMus]*x[i+nMus]);
}

void lagrangeResidualTorquesMultistage(double *u, double *g, void *){
    for(i=0; i<nPhases; ++i){
        for (unsigned int j=0; j<nTau; ++j){
            g[0]+=(u[i*(nMus+nTau) + nMus + j]*u[i*(nMus+nTau) + nMus + j]);
        }
    }
}

void lagrangeActivations( double *x, double *g, void *){
    g[0]=0;
    for (i=0; i<nMus; ++i)
        g[0]+=(x[i]*x[i]);
}

void lagrangeAccelerations( double *x, double *g, void *user_data){
    g[0]=0;
    double * rhs = new double[nQ + nQdot]; // memory management to check
    forwardDynamicsFromJointTorque(x, rhs, user_data);
    for (i=0; i<nQdot; ++i)
        g[0] += (rhs[i]*rhs[i]);
    delete[] rhs;
    std::cout<< g[0]<< std::endl;
}

void lagrangeTime( double *x, double *g, void *){
        g[0]=x[nQ+nQdot+nMus+nTau];
}

void mayerVelocity( double *x, double *g, void *){
    g[0]=0;
    for (i=0; i<nQdot; ++i)
        g[0]+=(x[i+nQ]*x[i+nQ]);
}

void mayerRHS( double *x, double *g, void *user_data){
    double * rhs = new double[nQ + nQdot];
    forwardDynamicsFromJointTorque(x, rhs, user_data);
    g[0] = 0;
    for (i = 0; i<nQ+nQdot; ++i)
        g[0] += (rhs[i]*rhs[i]);
    delete[] rhs;
}


