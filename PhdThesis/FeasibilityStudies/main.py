from enum import Enum

import numpy as np

from feasibility_studies import FatigueIntegrator, StudyConfiguration, FatigueModels, TargetFunctions, FatigueParameters, Integrator, CustomAnalysis, PlotOptions


class Study(Enum):
    # DEBUG OPTIONS
    XIA_ONLY = StudyConfiguration(
        fatigue_models=(
            FatigueModels.XIA(FatigueParameters(), integrator=Integrator.RK45, x0=(0, 1, 0), rms_indices=(0, 1, 2)),
            FatigueModels.XIA(FatigueParameters(), integrator=Integrator.RK4, x0=(0, 1, 0), rms_indices=(0, 1, 2)),
        ),
        t_end=30,
        fixed_target=0.2,
        target_function=TargetFunctions.TARGET_UP_TO_END,
        n_points=10000,  # 100000
        plot_options=PlotOptions(
            title="DEBUG: Modèle de Xia seulement", options=({"linestyle": "-"}, {"linestyle": "--"})
        ),
    )

    XIA_STABILIZED_ONLY = StudyConfiguration(
        fatigue_models=(
            FatigueModels.XIA_STABILIZED(FatigueParameters(), integrator=Integrator.RK45, x0=(0, 1, 0)),
        ),
        t_end=100,
        fixed_target=0.2,
        target_function=TargetFunctions.TARGET_UP_TO_END,
        n_points=4000,
        plot_options=PlotOptions(title="DEBUG: Modèle de Xia stabilisé seulement", options=({"linestyle": "-"}, )),
    )

    XIA_LONG = StudyConfiguration(
        fatigue_models=(
            FatigueModels.XIA(FatigueParameters(stabilization_factor=100), integrator=Integrator.RK45, x0=(0, 1, 0), rms_indices=(0, 1, 2)),
            FatigueModels.XIA_STABILIZED(FatigueParameters(stabilization_factor=100), integrator=Integrator.RK45, x0=(0, 1, 0), rms_indices=(0, 1, 2))
        ),
        t_end=3600,
        fixed_target=1,
        target_function=TargetFunctions.TARGET_RANDOM_PER_10SECONDS,
        n_points=100000,
        plot_options=PlotOptions(
            title="DEBUG: Xia uniquement sur 1 heure", options=({"linestyle": "-"}, {"linestyle": "--"})
        ),
    )

    # Actual studies from the thesis
    STUDY1_1_XIA_STABILIZED = StudyConfiguration(
        repeat=1,
        fatigue_models=(
            FatigueModels.XIA_STABILIZED(
                FatigueParameters(stabilization_factor=100),
                integrator=Integrator.RK45, x0=(0, 0.6, 0),
                custom_analyses=(
                    CustomAnalysis(
                        "First index with sum at 95%",
                        lambda result: np.where(np.sum(result.y, axis=0) > 0.95)[0][0]
                    ),
                    CustomAnalysis(
                        "First time with sum at 95%",
                        lambda result: result.t[np.where(np.sum(result.y, axis=0) > 0.95)[0][0]]
                    ),
                    CustomAnalysis(
                        "Fatigue at same time with sum at 95%",
                        lambda result: result.y[2, np.where(np.sum(result.y, axis=0) > 0.95)[0][0]]
                    ),
                    CustomAnalysis(
                        "Sum at the end of the trial",
                        lambda result: np.sum(result.y, axis=0)[-1]
                    ),
                ),
            ),
        ),
        t_end=0.1,
        fixed_target=0.8,
        target_function=TargetFunctions.TARGET_UP_TO_END,
        n_points=1000,
        plot_options=PlotOptions(
            title="Remplissages des bassins du modèle $3CC^S$ en fonction du temps",
            legend=("C(t)", "$3CC\ M_A$", "$3CC\ M_R$", "$3CC\ M_F$", "$\sum{3CC}$", "$3CC^S\ M_A$", "$3CC^S\ M_R$", "$3CC^S\ M_F$", "$\sum{3CC^S}$"),
            options=({"linestyle": "--"},)
        ),
    )

    STUDY1_2_XIA_VS_STABILIZED_GOOD_X0 = StudyConfiguration(
        repeat=1,
        fatigue_models=(
            FatigueModels.XIA(FatigueParameters(stabilization_factor=100), integrator=Integrator.RK45, x0=(0, 1, 0), rms_indices=(0, 1, 2)),
            FatigueModels.XIA_STABILIZED(FatigueParameters(stabilization_factor=100), integrator=Integrator.RK45, x0=(0, 1, 0), rms_indices=(0, 1, 2)),
        ),
        t_end=600,
        fixed_target=0.8,
        target_function=TargetFunctions.TARGET_UP_TO_END,
        n_points=1000,
        plot_options=PlotOptions(
            title="Remplissages des bassins des modèles $3CC$ et $3CC^S$ en fonction du temps",
            legend=("C(t)", "$3CC\ M_A$", "$3CC\ M_R$", "$3CC\ M_F$", "$\sum{3CC}$", "$3CC^S\ M_A$", "$3CC^S\ M_R$", "$3CC^S\ M_F$", "$\sum{3CC^S}$"),
            options=({"linestyle": "-"}, {"linestyle": "--"},)
        ),
    )

    STUDY1_3_XIA_VS_STABILIZED_BAD_X0 = StudyConfiguration(
        repeat=1,
        fatigue_models=(
            FatigueModels.XIA(FatigueParameters(stabilization_factor=100), integrator=Integrator.RK45, x0=(0, 0.6, 0), rms_indices=(0, 1, 2)),
            FatigueModels.XIA_STABILIZED(FatigueParameters(stabilization_factor=100), integrator=Integrator.RK45, x0=(0, 0.6, 0), rms_indices=(0, 1, 2))
        ),
        t_end=600,
        fixed_target=0.8,
        target_function=TargetFunctions.TARGET_UP_TO_END,
        n_points=1000,
        plot_options=PlotOptions(
            title="Remplissages des bassins des modèles $3CC$ et $3CC^S$ en fonction du temps",
            legend=("C(t)", "$3CC\ M_A$", "$3CC\ M_R$", "$3CC\ M_F$", "$\sum{3CC}$", "$3CC^S\ M_A$", "$3CC^S\ M_R$", "$3CC^S\ M_F$", "$\sum{3CC^S}$"),
            options=({"linestyle": "-"}, {"linestyle": "--"},),
            save_path="xia_vs_xiaStabilized_badStart.png"
        ),
    )


def main():
    # Define the study to perform
    study = Study.STUDY1_1_XIA_STABILIZED

    # Prepare and run the integrator
    runner = FatigueIntegrator(study.value)
    runner.perform()

    # Print some results
    runner.print_integration_time()
    runner.print_final_sum()
    if len(runner.study.fatigue_models) == 2:
        runner.print_rmse()
    runner.print_custom_analyses()
    runner.plot_results()


if __name__ == "__main__":
    main()
