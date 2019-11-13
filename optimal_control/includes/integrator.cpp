#include "integrator.h"
#include "biorbd.h"

AcadoIntegrator::AcadoIntegrator(biorbd::Model& m) :
    biorbd::rigidbody::Integrator(m),
    m_isKinematicsComputed(false)
{

}

void AcadoIntegrator::operator()(
        const state_type &x,
        state_type &dxdt,
        double t)
{
    std::cout << "coucou" << std::endl;
    biorbd::rigidbody::Integrator::operator ()(x, dxdt, t);
}

void AcadoIntegrator::integrateKinematics(
        const biorbd::rigidbody::GeneralizedCoordinates &Q,
        const biorbd::rigidbody::GeneralizedCoordinates &QDot,
        const biorbd::rigidbody::GeneralizedTorque &GeneralizedTorque,
        double t0,
        double tend,
        double timeStep)
{
    biorbd::utils::Vector v(static_cast<unsigned int>(Q.rows()+QDot.rows()));
    v << Q,QDot;
    integrate(v, GeneralizedTorque, t0, tend, timeStep); // vecteur, t0, tend, pas, effecteurs
    m_isKinematicsComputed = true;
}

void AcadoIntegrator::getIntegratedKinematics(
        unsigned int step,
        biorbd::rigidbody::GeneralizedCoordinates &Q,
        biorbd::rigidbody::GeneralizedCoordinates &QDot)
{
    // Si la cinématique n'a pas été mise à jour
    biorbd::utils::Error::check(
                m_isKinematicsComputed,
                "ComputeKinematics must be call before calling updateKinematics");
    const biorbd::utils::Vector& tp(getX(step));
    for (unsigned int i=0; i< static_cast<unsigned int>(tp.rows()/2); i++){
        Q(i) = tp(i);
        QDot(i) = tp(i+tp.rows()/2);
    }
}
unsigned int AcadoIntegrator::nbInterationStep() const
{
    return steps();
}
