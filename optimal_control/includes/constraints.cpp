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
    g[0] = x[1]+1.13;       // shoulder at 65° in abduction (Arm_RotX = -1.13)
    g[1] = x[2]-0.61;       // shoulder at 35° in flexion (Arm_RotZ = 0.61)
    g[2] = x[3]+0.35;       // shoulder at 20°  (Arm_RotY = -0.35)
    g[3] = x[4]-1.55;       // elbow at 110° (LowerArm1_RotZ = 1.55)
}

void ViolonDown( double *x, double *g, void * ){
    g[0] = x[1]+0.70;       // shoulder at 40° in  (Arm_RotX = -0.70)
    g[1] = x[2]-0.17;       // shoulder at 10° in flexion (Arm_RotZ = 0.17)
    g[2] = x[3];            // shoulder at 0°  (Arm_RotY = 0.0)
    g[3] = x[4]-0.61;       // elbow at 35° (LowerArm1_RotZ = 0.61)
}
