#include "constraints.h"


void StatesZero( double *x, double *g, void * ){
    for (unsigned int i =0; i<nQ + nQdot; ++i) {
        g[i] =  x[i];
    }
}

void VelocityZero( double *x, double *g, void * ){
    for (unsigned int i =0; i<nQdot; ++i) {
        g[i] =  x[nQ+i];
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

void ViolonUp( double *x, double *g, void * ){
    g[0] = x[1]+1.31;       // shoulder at 75° in abduction (Arm_RotX = -1.31)
    g[1] = x[2]-1.22;       // shoulder at 70° in flexion (Arm_RotZ = 1.22)
    g[2] = x[4]-1.92;       // elbow at 110° (LowerArm1_RotZ = 1.92)
}

void ViolonDown( double *x, double *g, void * ){
    g[0] = x[1]+0.87;       // shoulder at 75° in  (Arm_RotX = -0.87)
    g[1] = x[2];            //
    g[2] = x[4]-0.17;       // elbow at 110° (LowerArm1_RotZ = 0.17)
}
