#include "integrator.h"
#include "biorbd.h"

#include <boost/numeric/odeint.hpp>
#include "dynamics.h"

AcadoIntegrator::AcadoIntegrator(biorbd::Model& m) :
    biorbd::rigidbody::Integrator(m),
    m_isKinematicsComputed(false)
{
    m_lhs = new double[m_model->nbQ() + m_model->nbQdot()];
    m_rhs = new double[m_model->nbQ() + m_model->nbQdot()];
}


void AcadoIntegrator::operator()(
        const state_type &x,
        state_type &dxdt,
        double)
{
    for (unsigned int i=0; i<*m_nQ + *m_nQdot; i++){
        m_lhs[i] = x[i];
    }

    // Équation différentielle : x/xdot => xdot/xddot
    forwardDynamics_contact(m_lhs, m_rhs);

    // Faire sortir xdot/xddot
    for (unsigned int i=0; i<*m_nQ + *m_nQdot; i++){
        dxdt[i] = m_rhs[i];
    }

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

void AcadoIntegrator::launchIntegrate(
        state_type &x,
        double t0,
        double tend,
        double timeStep)
{
    // Choix de l'algorithme et intégration
    boost::numeric::odeint::runge_kutta4< state_type > stepper;
    *m_steps = static_cast<unsigned int>(
                boost::numeric::odeint::integrate_const(
                    stepper, *this, x, t0, tend, timeStep,
                    push_back_state_and_time( *m_x_vec , *m_times )));
}
