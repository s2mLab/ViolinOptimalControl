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
    def rmse_before_fatigue(studies):
        last_cycle = 200
        last_frame = last_cycle * studies.studies[0].nmpc.n_shooting_per_cycle

        for study, solution in zip(studies.studies, studies.solutions):
            ref_idx = study.rmse_index
            data_ref = studies.solutions[ref_idx][0].states["q"][:, :last_frame]
            data = solution[0].states["q"][:, :last_frame]

            e = data_ref - data
            se = e ** 2
            mse = np.sum(se, axis=1) / data_ref.shape[1]
            rmse = np.sqrt(mse)
            sum_rmse = np.sum(rmse)
            print(f"\tRMSE pour {study.name}")
            print(f"\t\tLa somme des RMSE jusqu'au cycle {last_cycle} est de : {sum_rmse:0.3f}")

    @staticmethod
    def rmse_after_fatigue(studies):
        first_cycle = 550
        first_frame = first_cycle * studies.studies[0].nmpc.n_shooting_per_cycle

        for study, solution in zip(studies.studies, studies.solutions):
            ref_idx = study.rmse_index
            data_ref = studies.solutions[ref_idx][0].states["q"][:, first_frame:]
            data = solution[0].states["q"][:, first_frame:]

            e = data_ref - data
            se = e ** 2
            mse = np.sum(se, axis=1) / data_ref.shape[1]
            rmse = np.sqrt(mse)
            sum_rmse = np.sum(rmse)
            print(f"\tRMSE pour {study.name}")
            print(f"\t\tLa valeur minimale des RMSE à partir du cycle {first_cycle} est de : {min(rmse[:-1]):0.3f}")
            print(f"\t\tLa somme des RMSE à partir du cycle {first_cycle} est de : {sum_rmse:0.3f}")

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
    RMSE_BEFORE_FATIGUE = CustomAnalysesImplementation.rmse_before_fatigue
    RMSE_AFTER_FATIGUE = CustomAnalysesImplementation.rmse_after_fatigue
    OBJECTIVE_FUNCTION_AT_50_AND_550 = CustomAnalysesImplementation.objective_function_at_50_and_550


class CustomAnalyses:
    def __init__(self, analyses: tuple[CustomAnalysesFcn, ...]):
        self.analyses: tuple[Union[Callable, CustomAnalysesFcn], ...] = analyses

    def perform(self, studies):
        for analysis in self.analyses:
            analysis(studies)
