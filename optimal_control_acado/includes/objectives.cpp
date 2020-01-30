#include "objectives.h"

static unsigned int i;

void residualTorquesSquare( double *u, double *g, void *){
    g[0]=0;
    for (i=0; i<nTau; ++i)
        g[0] += u[i+nMus] * u[i+nMus];
}

void muscleActivationsSquare( double *x, double *g, void *){
    g[0]=0;
    for (i=0; i<nMus; ++i)
        g[0] += x[i] * x[i];
}

void stringToPlayObjective(double *x, double *g, void *user_data){
    dispatchQ(x);
    m.UpdateKinematicsCustom(&Q);
    const biorbd::rigidbody::NodeSegment& markerBowFrog(m.marker(Q, static_cast<unsigned int*>(user_data)[0], false, false));
    const biorbd::rigidbody::NodeSegment& markerBowTip(m.marker(Q, static_cast<unsigned int*>(user_data)[1], false, false));
    const biorbd::rigidbody::NodeSegment& markerViolinStringProx(m.marker(Q, static_cast<unsigned int*>(user_data)[2], false, false));
    const biorbd::rigidbody::NodeSegment& markerViolinStringDist(m.marker(Q, static_cast<unsigned int*>(user_data)[3], false, false));

    biorbd::utils::Vector3d bowAxis( (markerBowTip - markerBowFrog) );
    biorbd::utils::Vector3d violinAxis( (markerViolinStringDist - markerViolinStringProx) );
    bowAxis.normalize();
    violinAxis.normalize();

    g[0] = (1 - bowAxis.dot(violinAxis));
}

void accelerationsObjective( double *x, double *g, void *user_data){
    double * rhs = new double[nQ + nQdot]; // memory management to check
    forwardDynamics_noContact(x, rhs, user_data);

    g[0]=0;
    for (i=0; i<nQdot; ++i)
        g[0] += (rhs[i]*rhs[i]);
    delete[] rhs;
}
