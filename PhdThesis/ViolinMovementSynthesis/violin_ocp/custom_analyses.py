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
    def rmse(studies, first_cycle: int = 0, last_cycle: int = None):
        if last_cycle is None:
            last_cycle = studies.studies[0].n_cycles_total
        first_frame = first_cycle * studies.studies[0].nmpc.n_shooting_per_cycle
        last_frame = last_cycle * studies.studies[0].nmpc.n_shooting_per_cycle

        for study, solution in zip(studies.studies, studies.solutions):
            ref_idx = study.rmse_index
            data_ref = studies.solutions[ref_idx][0].states["q"][:, first_frame:last_frame]
            data = solution[0].states["q"][:, first_frame:last_frame]

            e = data_ref - data
            se = e ** 2
            mse = np.sum(se, axis=1) / data_ref.shape[1]
            rmse = np.sqrt(mse)
            sum_rmse = np.sum(rmse)
            print(f"\tRMSE pour {study.name}")
            print(f"\t\tLa somme des RMSE des cycles {first_cycle} à {last_cycle} est de : {sum_rmse:0.3f}")

    @staticmethod
    def objective_function(studies, cycle):
        for study in studies.studies:
            f_path = f"{studies.prepare_and_get_results_dir()}/{study.save_name}_iterations/iteration_{cycle:04d}.bo"
            ocp, sol = study.nmpc.load(f_path)
            print(f"Fonctions objectifs pour {study.save_name} au cycle {cycle}")
            sol.print(CostType.OBJECTIVES)


class CustomAnalysesFcn(Enum):
    PRINT_NUMBER_OF_ITERATIONS = CustomAnalysesImplementation.print_number_of_iterations
    RMSE = CustomAnalysesImplementation.rmse
    OBJECTIVE_FUNCTION = CustomAnalysesImplementation.objective_function


class CustomAnalysesOption:
    def __init__(self, analysis=Union[Callable, CustomAnalysesFcn], **params):
        self.analysis = analysis
        self.params = params


class CustomAnalyses:
    def __init__(self, analyses: tuple[CustomAnalysesOption, ...]):
        self.analyses: tuple[CustomAnalysesOption, ...] = analyses

    def perform(self, studies):
        for analysis in self.analyses:
            analysis.analysis(studies, **analysis.params)
