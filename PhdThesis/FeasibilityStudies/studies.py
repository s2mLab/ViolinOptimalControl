from enum import Enum

import numpy as np

from feasibility_studies import (
    StudyConfiguration,
    FatigueModels,
    TargetFunctions,
    FatigueParameters,
    Integrator,
    CustomAnalysis,
    PlotOptions,
)


class Study(Enum):
    # DEBUG OPTIONS
    DEBUG_XIA_ONLY = StudyConfiguration(
        name="DEBUG_XIA_ONLY",
        fatigue_models=(
            FatigueModels.XIA(
                FatigueParameters(),
                integrator=Integrator.RK45,
                x0=(0, 1, 0),
                rms_indices=(0, 1, 2),
                colors=("tab:green", "tab:orange", "tab:red"),
            ),
            FatigueModels.XIA(
                FatigueParameters(F=1),
                integrator=Integrator.RK45,
                x0=(0, 1, 0),
                rms_indices=(0, 1, 2),
                colors=("tab:green", "tab:orange", "tab:red"),
            ),
        ),
        t_end=300,
        fixed_target=0.1,
        target_function=TargetFunctions.TARGET_UP_TO_END,
        n_points=10000,  # 100000
        plot_options=PlotOptions(
            title="DEBUG: Modèle de Xia seulement",
            options=({"linestyle": "-"}, {"linestyle": "--"}),
        ),
    )

    DEBUG_XIA_STABILIZED_ONLY = StudyConfiguration(
        name="DEBUG_XIA_STABILIZED_ONLY",
        fatigue_models=(
            FatigueModels.XIA_STABILIZED(
                FatigueParameters(),
                integrator=Integrator.RK45,
                x0=(0, 1, 0),
                colors=("tab:green", "tab:orange", "tab:red"),
            ),
        ),
        t_end=100,
        fixed_target=0.2,
        target_function=TargetFunctions.TARGET_UP_TO_END,
        n_points=4000,
        plot_options=PlotOptions(
            title="DEBUG: Modèle de Xia stabilisé seulement",
            options=({"linestyle": "-"},),
        ),
    )

    DEBUG_XIA_LONG = StudyConfiguration(
        name="DEBUG_XIA_LONG",
        fatigue_models=(
            FatigueModels.XIA(
                FatigueParameters(stabilization_factor=100),
                integrator=Integrator.RK45,
                x0=(0, 1, 0),
                rms_indices=(0, 1, 2),
                colors=("tab:green", "tab:orange", "tab:red"),
            ),
            FatigueModels.XIA_STABILIZED(
                FatigueParameters(stabilization_factor=100),
                integrator=Integrator.RK45,
                x0=(0, 1, 0),
                rms_indices=(0, 1, 2),
                colors=("tab:green", "tab:orange", "tab:red"),
            ),
        ),
        t_end=3600,
        fixed_target=1,
        target_function=TargetFunctions.TARGET_RANDOM_PER_10SECONDS,
        n_points=100000,
        plot_options=PlotOptions(
            title="DEBUG: Xia uniquement sur 1 heure",
            options=({"linestyle": "-"}, {"linestyle": "--"}),
        ),
    )

    DEBUG_MICHAUD_ONLY = StudyConfiguration(
        name="DEBUG_MICHAUD_ONLY",
        fatigue_models=(
            FatigueModels.MICHAUD(
                FatigueParameters(),
                integrator=Integrator.RK45,
                x0=(0, 1, 0, 0),
                rms_indices=(
                    0,
                    1,
                    2,
                    3,
                ),
                colors=("tab:green", "tab:orange", "tab:red", "tab:gray"),
            ),
            FatigueModels.MICHAUD(
                FatigueParameters(effort_factor=0.1),
                integrator=Integrator.RK45,
                x0=(0, 1, 0, 0),
                rms_indices=(
                    0,
                    1,
                    2,
                    3,
                ),
                colors=("tab:green", "tab:orange", "tab:red", "tab:gray"),
            ),
        ),
        t_end=600,
        fixed_target=0.8,
        target_function=TargetFunctions.TARGET_UP_TO_END,
        n_points=10000,  # 100000
        plot_options=PlotOptions(
            title="DEBUG: Modèle de Michaud seulement", options=({"linestyle": "-"}, {"linestyle": "--"})
        ),
    )

    DEBU_EP_ONLY = StudyConfiguration(
        name="DEBU_EP_ONLY",
        fatigue_models=(
            FatigueModels.EFFORT_PERCEPTION(
                FatigueParameters(),
                integrator=Integrator.RK45,
                x0=(0,),
                rms_indices=(0,),
                colors=("tab:gray",),
            ),
        ),
        t_end=600,
        fixed_target=0.8,
        target_function=TargetFunctions.TARGET_UP_TO_END,
        n_points=10000,  # 100000
        plot_options=PlotOptions(
            title="DEBUG: Modèle de Pe seulement", options=({"linestyle": "-"}, {"linestyle": "--"})
        ),
    )

    # Actual studies from the thesis
    STUDY1_1_XIA_STABILIZED = StudyConfiguration(
        name="STUDY1_1_XIA_STABILIZED",
        repeat=1,
        fatigue_models=(
            FatigueModels.XIA_STABILIZED(
                FatigueParameters(stabilization_factor=200),
                integrator=Integrator.RK45,
                x0=(0, 0.6, 0),
                colors=("tab:green", "tab:orange", "tab:red"),
                custom_analyses=(
                    CustomAnalysis(
                        "First index with sum at 95%", lambda result: np.where(np.sum(result.y, axis=0) > 0.95)[0][0]
                    ),
                    CustomAnalysis(
                        "First time with sum at 95%",
                        lambda result: result.t[np.where(np.sum(result.y, axis=0) > 0.95)[0][0]],
                    ),
                    CustomAnalysis(
                        "Fatigue at same time with sum at 95%",
                        lambda result: result.y[2, np.where(np.sum(result.y, axis=0) > 0.95)[0][0]],
                    ),
                    CustomAnalysis("Sum at the end of the trial", lambda result: np.sum(result.y, axis=0)[-1]),
                ),
            ),
            FatigueModels.XIA_STABILIZED(
                FatigueParameters(stabilization_factor=100),
                integrator=Integrator.RK45,
                x0=(0, 0.6, 0),
                colors=("tab:green", "tab:orange", "tab:red"),
                custom_analyses=(
                    CustomAnalysis(
                        "First index with sum at 95%", lambda result: np.where(np.sum(result.y, axis=0) > 0.95)[0][0]
                    ),
                    CustomAnalysis(
                        "First time with sum at 95%",
                        lambda result: result.t[np.where(np.sum(result.y, axis=0) > 0.95)[0][0]],
                    ),
                    CustomAnalysis(
                        "Fatigue at same time with sum at 95%",
                        lambda result: result.y[2, np.where(np.sum(result.y, axis=0) > 0.95)[0][0]],
                    ),
                    CustomAnalysis("Sum at the end of the trial", lambda result: np.sum(result.y, axis=0)[-1]),
                ),
            ),
            FatigueModels.XIA_STABILIZED(
                FatigueParameters(stabilization_factor=50),
                integrator=Integrator.RK45,
                x0=(0, 0.6, 0),
                colors=("tab:green", "tab:orange", "tab:red"),
                custom_analyses=(
                    CustomAnalysis(
                        "First index with sum at 95%", lambda result: np.where(np.sum(result.y, axis=0) > 0.95)[0][0]
                    ),
                    CustomAnalysis(
                        "First time with sum at 95%",
                        lambda result: result.t[np.where(np.sum(result.y, axis=0) > 0.95)[0][0]],
                    ),
                    CustomAnalysis(
                        "Fatigue at same time with sum at 95%",
                        lambda result: result.y[2, np.where(np.sum(result.y, axis=0) > 0.95)[0][0]],
                    ),
                    CustomAnalysis("Sum at the end of the trial", lambda result: np.sum(result.y[:, -1])),
                ),
            ),
            FatigueModels.XIA_STABILIZED(
                FatigueParameters(stabilization_factor=0),
                integrator=Integrator.RK45,
                x0=(0, 0.6, 0),
                colors=("tab:green", "tab:orange", "tab:red"),
                custom_analyses=(
                    CustomAnalysis("Sum at the end of the trial", lambda result: np.sum(result.y[:, -1])),
                ),
            ),
        ),
        t_end=0.1,
        fixed_target=0.8,
        target_function=TargetFunctions.TARGET_UP_TO_END,
        n_points=1000,
        plot_options=PlotOptions(
            title="Remplissages des bassins du modèle $3CC^S$ en fonction du temps",
            legend=(
                "_",
                "_",
                "_",
                "_",
                "$S = 200$",
                "_",
                "_",
                "_",
                "$S = 100$",
                "_",
                "_",
                "_",
                "$S = 50$",
                "_",
                "_",
                "_",
                "$S = 0$",
            ),
            supplementary_legend=("Cible", "$M_A$", "$M_R$", "$M_F$", "$\sum{}$"),
            options=({"linestyle": "-"}, {"linestyle": "--"}, {"linestyle": "-."}, {"linestyle": ":"}),
            save_path="xiaStabilized_short.png",
        ),
        common_custom_analyses=(
            CustomAnalysis("Sum of components at the final index", lambda results: np.sum(results.y[:, -1], axis=0)),
        ),
    )

    STUDY1_2_XIA_VS_STABILIZED_GOOD_X0 = StudyConfiguration(
        name="STUDY1_2_XIA_VS_STABILIZED_GOOD_X0",
        repeat=10,
        fatigue_models=(
            FatigueModels.XIA_STABILIZED(
                FatigueParameters(stabilization_factor=100),
                integrator=Integrator.RK45,
                x0=(0, 1, 0),
                colors=("tab:green", "tab:orange", "tab:red"),
                rms_indices=(0, 1, 2),
            ),
            FatigueModels.XIA(
                FatigueParameters(stabilization_factor=100),
                integrator=Integrator.RK45,
                x0=(0, 1, 0),
                colors=("tab:green", "tab:orange", "tab:red"),
                rms_indices=(0, 1, 2),
            ),
        ),
        t_end=600,
        fixed_target=0.8,
        target_function=TargetFunctions.TARGET_UP_TO_END,
        n_points=1000,
        plot_options=PlotOptions(
            title="Remplissages des bassins des modèles $3CC$ et $3CC^S$ en fonction du temps",
            legend=("_", "_", "_", "_", "$3CC^S$", "_", "_", "_", "$3CC$"),
            supplementary_legend=("Cible", "$M_A$", "$M_R$", "$M_F$", "$\sum{}$"),
            options=(
                {"linestyle": "-"},
                {"linestyle": "--"},
            ),
            save_path="xia_vs_xiaStabilized_goodStart.png",
        ),
        common_custom_analyses=(
            CustomAnalysis("Sum of components at the final index", lambda results: np.sum(results.y[:, -1], axis=0)),
            CustomAnalysis("Fatigue at final node", lambda result: result.y[2, -1]),
        ),
    )

    STUDY1_3_XIA_VS_STABILIZED_BAD_X0 = StudyConfiguration(
        name="STUDY1_3_XIA_VS_STABILIZED_BAD_X0",
        repeat=10,
        fatigue_models=(
            FatigueModels.XIA_STABILIZED(
                FatigueParameters(stabilization_factor=100),
                integrator=Integrator.RK45,
                x0=(0, 0.6, 0),
                colors=("tab:green", "tab:orange", "tab:red"),
                rms_indices=(0, 1, 2),
            ),
            FatigueModels.XIA(
                FatigueParameters(stabilization_factor=100),
                integrator=Integrator.RK45,
                x0=(0, 0.6, 0),
                colors=("tab:green", "tab:orange", "tab:red"),
                rms_indices=(0, 1, 2),
            ),
        ),
        t_end=600,
        fixed_target=0.8,
        target_function=TargetFunctions.TARGET_UP_TO_END,
        n_points=1000,
        plot_options=PlotOptions(
            title="Remplissages des bassins des modèles $3CC$ et $3CC^S$ en fonction du temps",
            legend=("_", "_", "_", "_", "$3CC^S$", "_", "_", "_", "$3CC$"),
            supplementary_legend=("Cible", "$M_A$", "$M_R$", "$M_F$", "$\sum{}$"),
            options=(
                {"linestyle": "-"},
                {"linestyle": "--"},
            ),
            save_path="xia_vs_xiaStabilized_badStart.png",
        ),
        common_custom_analyses=(
            CustomAnalysis("Sum of components at the final index", lambda results: np.sum(results.y[:, -1], axis=0)),
            CustomAnalysis("Fatigue at final node", lambda result: result.y[2, -1]),
        ),
    )

    STUDY1_4_XIA_STABILIZED_FATIGUE_NEGATIVE = StudyConfiguration(
        name="STUDY1_4_XIA_STABILIZED_FATIGUE_NEGATIVE",
        repeat=1,
        fatigue_models=(
            FatigueModels.XIA_STABILIZED(
                FatigueParameters(),
                integrator=Integrator.RK45,
                x0=(1, 1, 0),
                colors=("tab:green", "tab:orange", "tab:red"),
                rms_indices=(0, 1, 2),
            ),
            FatigueModels.XIA(
                FatigueParameters(),
                integrator=Integrator.RK45,
                x0=(1, 1, 0),
                colors=("tab:green", "tab:orange", "tab:red"),
                rms_indices=(0, 1, 2),
            ),
        ),
        t_end=0.5,
        fixed_target=0.8,
        target_function=TargetFunctions.TARGET_UP_TO_END,
        n_points=1000,
        plot_options=PlotOptions(
            title="Remplissages des bassins des modèles $3CC$ et $3CC^S$ en fonction du temps",
            legend=("_", "_", "_", "_", "$3CC^S$", "_", "_", "_", "$3CC$"),
            supplementary_legend=("Cible", "$M_A$", "$M_R$", "$M_F$", "$\sum{}$"),
            options=(
                {"linestyle": "-"},
                {"linestyle": "--"},
            ),
            save_path="xiaStabilized_fatigueNegative.png",
        ),
        common_custom_analyses=(
            CustomAnalysis("Sum of components at the final index", lambda results: np.sum(results.y[:, -1], axis=0)),
            CustomAnalysis("Fatigue at final node", lambda result: result.y[2, -1]),
        ),
    )

    STUDY2_1_MICHAUD_LONG = StudyConfiguration(
        name="STUDY2_1_MICHAUD_LONG",
        fatigue_models=(
            FatigueModels.MICHAUD(
                FatigueParameters(),
                integrator=Integrator.RK45,
                x0=(0, 1, 0, 0),
                colors=("tab:green", "tab:orange", "tab:red", "tab:gray"),
                rms_indices=(),
            ),
            FatigueModels.XIA(
                FatigueParameters(),
                integrator=Integrator.RK45,
                x0=(0, 1, 0),
                colors=("tab:green", "tab:orange", "tab:red"),
                rms_indices=(),
            ),
        ),
        t_end=600,
        fixed_target=0.4,
        target_function=TargetFunctions.TARGET_UP_TO_END,
        n_points=10000,  # 100000
        plot_options=PlotOptions(
            title="Remplissages des bassins des modèles $3CC$ et $4CC$ en fonction du temps",
            legend=("_", "_", "_", "_", "_", "$4CC$", "_", "_", "_", "$3CC$"),
            supplementary_legend=("Cible", "$M_A$", "$M_R$", "$M_F$", "$M_E$", "$\sum{}$"),
            options=({"linestyle": "-"}, {"linestyle": "--"}),
            save_path="michaud_vs_xia_long.png",
        ),
    )

    STUDY2_2_MICHAUD_VELOCITY_COMPARISON = StudyConfiguration(
        name="STUDY2_2_MICHAUD_VELOCITY_COMPARISON",
        fatigue_models=(
            FatigueModels.MICHAUD(
                FatigueParameters(effort_factor=0.0075, effort_threshold=0.5),
                integrator=Integrator.RK45,
                x0=(0, 1, 0, 0),
                colors=("tab:green", "tab:orange", "tab:red", "tab:gray"),
                rms_indices=(),
            ),
            FatigueModels.MICHAUD(
                FatigueParameters(effort_factor=0.0050, effort_threshold=0.5),
                integrator=Integrator.RK45,
                x0=(0, 1, 0, 0),
                colors=("tab:green", "tab:orange", "tab:red", "tab:gray"),
                rms_indices=(),
            ),
            FatigueModels.MICHAUD(
                FatigueParameters(effort_factor=0.0025, effort_threshold=0.5),
                integrator=Integrator.RK45,
                x0=(0, 1, 0, 0),
                colors=("tab:green", "tab:orange", "tab:red", "tab:gray"),
                rms_indices=(),
            ),
            FatigueModels.MICHAUD(
                FatigueParameters(effort_factor=0.0000, effort_threshold=0.5),
                integrator=Integrator.RK45,
                x0=(0, 1, 0, 0),
                colors=("tab:green", "tab:orange", "tab:red", "tab:gray"),
                rms_indices=(),
            ),
        ),
        t_end=100,
        fixed_target=(0.9, 0.1),
        target_function=TargetFunctions.TARGET1_UP_TO_MID_THEN_TARGET2,
        n_points=10000,  # 100000
        plot_options=PlotOptions(
            title="Remplissages des bassins du modèle $4CC$ en fonction du temps",
            legend=(
                "_",
                "_",
                "_",
                "_",
                "_",
                "$E = 0.0075$",
                "_",
                "_",
                "_",
                "_",
                "$E = 0.0050$",
                "_",
                "_",
                "_",
                "_",
                "$E = 0.0025$",
                "_",
                "_",
                "_",
                "_",
                "$E = 0.0000$",
            ),
            supplementary_legend=("Cible", "$M_A$", "$M_R$", "$M_F$", "$M_E$", "$\sum{}$"),
            options=(
                {"linestyle": "-"},
                {"linestyle": "--"},
                {"linestyle": "-."},
                {"linestyle": ":"},
            ),
            save_path="michaud_velocity.png",
        ),
        common_custom_analyses=(
            CustomAnalysis(
                "Effort perception level at mid point", lambda result: result.y[3, int(result.y.shape[1] // 2)]
            ),
            CustomAnalysis("Effort perception level at end point", lambda result: result.y[3, -1]),
            CustomAnalysis("Rested level at end point", lambda result: result.y[1, -1]),
        ),
    )

    STUDY2_3_MICHAUD_THRESHOLD_COMPARISON = StudyConfiguration(
        name="STUDY2_3_MICHAUD_THRESHOLD_COMPARISON",
        fatigue_models=(
            FatigueModels.MICHAUD(
                FatigueParameters(effort_threshold=0.75),
                integrator=Integrator.RK45,
                x0=(0, 1, 0, 0),
                colors=("tab:green", "tab:orange", "tab:red", "tab:gray"),
            ),
            FatigueModels.MICHAUD(
                FatigueParameters(effort_threshold=0.50),
                integrator=Integrator.RK45,
                x0=(0, 1, 0, 0),
                colors=("tab:green", "tab:orange", "tab:red", "tab:gray"),
            ),
            FatigueModels.MICHAUD(
                FatigueParameters(effort_threshold=0.25),
                integrator=Integrator.RK45,
                x0=(0, 1, 0, 0),
                colors=("tab:green", "tab:orange", "tab:red", "tab:gray"),
            ),
            FatigueModels.MICHAUD(
                FatigueParameters(effort_threshold=0.0001),
                integrator=Integrator.RK45,
                x0=(0, 1, 0, 0),
                colors=("tab:green", "tab:orange", "tab:red", "tab:gray"),
            ),
        ),
        t_end=100,
        fixed_target=(0.9, 0.1),
        target_function=TargetFunctions.TARGET1_UP_TO_MID_THEN_TARGET2,
        n_points=10000,  # 100000
        plot_options=PlotOptions(
            title="Remplissages des bassins du modèle $4CC$ en fonction du temps",
            legend=(
                "_",
                "_",
                "_",
                "_",
                "_",
                "$E_{seuil} = 75\%$",
                "_",
                "_",
                "_",
                "_",
                "$E_{seuil} = 50\%$",
                "_",
                "_",
                "_",
                "_",
                "$E_{seuil} = 25\%$",
                "_",
                "_",
                "_",
                "_",
                "$E_{seuil} \\approx 0\%$",
            ),
            supplementary_legend=("Cible", "$M_A$", "$M_R$", "$M_F$", "$M_E$", "$\sum{}$"),
            options=(
                {"linestyle": "-"},
                {"linestyle": "--"},
                {"linestyle": "-."},
                {"linestyle": ":"},
            ),
            save_path="michaud_threshold.png",
        ),
        common_custom_analyses=(
            CustomAnalysis(
                "Effort perception level at mid point", lambda result: result.y[3, int(result.y.shape[1] // 2)]
            ),
            CustomAnalysis("Effort perception level at end point", lambda result: result.y[3, -1]),
            CustomAnalysis(
                "Difference effort perception level at mid vs end point",
                lambda result: result.y[3, int(result.y.shape[1] // 2)] - result.y[3, -1],
            ),
        ),
    )

    STUDY2_4_MICHAUD_VS_XIA = StudyConfiguration(
        name="STUDY2_4_MICHAUD_VS_XIA",
        repeat=10,
        fatigue_models=(
            FatigueModels.MICHAUD(
                FatigueParameters(effort_threshold=0.5, F=0.01 / 4),
                integrator=Integrator.RK45,
                x0=(0, 1, 0, 0),
                colors=("tab:green", "tab:orange", "tab:red", "tab:gray"),
                rms_indices=(0, 1, 2),
                custom_analyses=(
                    CustomAnalysis(
                        "Effort perception at mid point", lambda result: result.y[3, int(result.y.shape[1] // 2 + 5)]
                    ),
                    CustomAnalysis("Effort perception at end point", lambda result: result.y[3, -1]),
                ),
            ),
            FatigueModels.XIA(
                FatigueParameters(F=0.01),
                integrator=Integrator.RK45,
                x0=(0, 1, 0),
                rms_indices=(0, 1, 2),
            ),
        ),
        t_end=100,
        fixed_target=(0.9, 0.1),
        target_function=TargetFunctions.TARGET1_UP_TO_MID_THEN_TARGET2,
        n_points=10000,  # 100000
        plot_options=PlotOptions(
            title="Remplissages des bassins des modèles $3CC$ et $4CC$ en fonction du temps",
            legend=("_", "_", "_", "_", "_", "$4CC$", "_", "_", "_", "$3CC$"),
            supplementary_legend=(
                "Cible",
                "$M_A$",
                "$M_R$",
                "$M_F$",
                "$M_E$",
                "$\sum{}$",
                "_",
                "_",
                "_",
            ),
            options=({"linestyle": "-"}, {"linestyle": "--"}),
            save_path="michaud_vs_xia.png",
        ),
        common_custom_analyses=(
            CustomAnalysis("Rested level at mid point", lambda result: result.y[1, int(result.y.shape[1] // 2 + 5)]),
            CustomAnalysis("Rested level at end point", lambda result: result.y[1, -1]),
            CustomAnalysis("Fatigue level at mid point", lambda result: result.y[2, int(result.y.shape[1] // 2 + 5)]),
            CustomAnalysis("Fatigue level at end point", lambda result: result.y[2, -1]),
        ),
    )

    STUDY3_0_EFFORT_VS_MICHAUD_LONG = StudyConfiguration(
        name="STUDY3_0_EFFORT_VS_MICHAUD_LONG",
        repeat=1,
        fatigue_models=(
            FatigueModels.EFFORT_PERCEPTION(
                FatigueParameters(),
                integrator=Integrator.RK45,
                x0=(0,),
                colors=("tab:gray",),
                rms_indices=(),
                print_sum=False,
            ),
            FatigueModels.MICHAUD(
                FatigueParameters(),
                integrator=Integrator.RK45,
                x0=(
                    0,
                    1,
                    0,
                    0,
                ),
                colors=("tab:green", "tab:orange", "tab:red", "tab:gray"),
                rms_indices=(),
            ),
        ),
        t_end=600,
        fixed_target=(0.9, 0.1),
        target_function=TargetFunctions.TARGET1_UP_TO_MID_THEN_TARGET2,
        n_points=10000,  # 100000
        plot_options=PlotOptions(
            title="Remplissages des bassins des modèles $PE$ et $4CC$ en fonction du temps",
            legend=("_", "PE", "_", "_", "_", "4CC"),
            supplementary_legend=("Cible", "$M_E$", "$M_A$", "$M_R$", "$M_F$", "_", "$\sum{}$"),
            options=({"linestyle": "-"}, {"linestyle": "--"}),
            save_path="effortPerception_vs_michaud.png",
        ),
    )
