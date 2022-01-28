from enum import Enum
from typing import Union, Callable

from bioptim import CostType
import numpy as np


class CustomAnalysesImplementation:
    @staticmethod
    def print_number_of_iterations(studies):
        print("Nombre d'itérations pour :")
        for study, (solution, all_iterations) in zip(studies.studies, studies.solutions):
            print(f"\t{study.name} : {[i.iterations for i in all_iterations]}")
        print("")

    @staticmethod
    def rmse_up_to_cycle_500(studies):
        last_cycle = 500
        idx = 1
        ref_idx = studies.studies[idx].rmse_index

        last_frame = last_cycle * studies.studies[0].nmpc.n_shooting_per_cycle
        data_ref = studies.solutions[ref_idx][0].states["q"][:, :last_frame]
        data = studies.solutions[idx][0].states["q"][:, :last_frame]

        e = data_ref - data
        se = e ** 2
        mse = np.sum(se, axis=1) / data_ref.shape[1]
        rmse = np.sqrt(mse)
        sum_rmse = np.sum(rmse)
        print(f"La somme des RMSE jusqu'au cycle {last_cycle} est de : {sum_rmse:0.3f}")

    @staticmethod
    def rmse_from_cycle_500(studies):
        first_cycle = 500
        idx = 1
        ref_idx = studies.studies[idx].rmse_index

        first_frame = first_cycle * studies.studies[0].nmpc.n_shooting_per_cycle
        data_ref = studies.solutions[ref_idx][0].states["q"][:, first_frame:]
        data = studies.solutions[idx][0].states["q"][:, first_frame:]

        e = data_ref - data
        se = e ** 2
        mse = np.sum(se, axis=1) / data_ref.shape[1]
        rmse = np.sqrt(mse)
        sum_rmse = np.sum(rmse)
        print(f"La valeur minimale des RMSE à partir du cycle {first_cycle} est de : {min(rmse[:-1]):0.3f}")
        print(f"La somme des RMSE à partir du cycle {first_cycle} est de : {sum_rmse:0.3f}")

    @staticmethod
    def objective_function_at_50_and_550(studies):
        cycles_to_print = (50, 550)

        for study in studies.studies:
            for i in cycles_to_print:
                f_path = f"{studies.prepare_and_get_results_dir()}/{study.save_name}_iterations/iteration_{i:04d}.bo"
                ocp, sol = study.nmpc.load(f_path)
                print(f"Fonctions objectifs pour {study.save_name} au cycle {i}")
                sol.print(CostType.OBJECTIVES)


class CustomAnalysesFcn(Enum):
    PRINT_NUMBER_OF_ITERATIONS = CustomAnalysesImplementation.print_number_of_iterations
    RMSE_UP_TO_CYCLE_450 = CustomAnalysesImplementation.rmse_up_to_cycle_500
    RMSE_FROM_CYCLE_510 = CustomAnalysesImplementation.rmse_from_cycle_500
    OBJECTIVE_FUNCTION_AT_50_AND_550 = CustomAnalysesImplementation.objective_function_at_50_and_550


class CustomAnalyses:
    def __init__(self, analyses: tuple[CustomAnalysesFcn, ...]):
        self.analyses: tuple[Union[Callable, CustomAnalysesFcn], ...] = analyses

    def perform(self, studies):
        for analysis in self.analyses:
            analysis(studies)
