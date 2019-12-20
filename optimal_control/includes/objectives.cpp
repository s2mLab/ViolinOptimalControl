#include "objectives.h"

static unsigned int i;

void residualTorquesSquare( double *u, double *g, void *){
    g[0]=0;
    for (i=0; i<nTau; ++i)
        g[0]+=(u[i+nMus]*u[i+nMus]);
}

void muscleActivationsSquare( double *x, double *g, void *){
    g[0]=0;
    for (i=0; i<nMus; ++i)
        g[0]+=(x[i]*x[i]);
}

void bowDirectionAgainstViolin(double *x, double *g, void *user_data){
    dispatchQ(x);
    m.UpdateKinematicsCustom(&Q);
    const biorbd::rigidbody::NodeSegment& markerBowDist(m.marker(Q, static_cast<unsigned int*>(user_data)[1], false, false));
    const biorbd::rigidbody::NodeSegment& markerBowProx(m.marker(Q, static_cast<unsigned int*>(user_data)[0], false, false));
    const biorbd::rigidbody::NodeSegment& markerViolinDist(m.marker(Q, static_cast<unsigned int*>(user_data)[3], false, false));
    const biorbd::rigidbody::NodeSegment& markerViolinProx(m.marker(Q, static_cast<unsigned int*>(user_data)[2], false, false));

    biorbd::utils::Vector3d BowAxe( (markerBowDist - markerBowProx) );
    biorbd::utils::Vector3d violinAxe( (markerViolinDist - markerViolinProx) );
    BowAxe.normalize();
    violinAxe.normalize();

    *g = (1 - BowAxe.dot(violinAxe))*100;
}

void lagrangeAccelerations( double *x, double *g, void *user_data){
    g[0]=0;
    double * rhs = new double[nQ + nQdot]; // memory management to check
    forwardDynamics_noContact(x, rhs, user_data);
    for (i=0; i<nQdot; ++i)
        g[0] += (rhs[i]*rhs[i]);
    delete[] rhs;
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
    forwardDynamics_noContact(x, rhs, user_data);
    g[0] = 0;
    for (i = 0; i<nQ+nQdot; ++i)
        g[0] += (rhs[i]*rhs[i]);
    delete[] rhs;
}


