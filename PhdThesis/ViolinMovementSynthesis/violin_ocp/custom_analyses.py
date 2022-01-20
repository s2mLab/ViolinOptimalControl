from enum import Enum
from typing import Union, Callable


class CustomAnalysesImplementation:
    @staticmethod
    def print_number_of_iterations(studies):
        print("Nombre d'itérations pour :")
        for study, (solution, all_iterations) in zip(studies.studies, studies.solutions):
            print(f"\t{study.name} : {[i.iterations for i in all_iterations]}")
        print("")


class CustomAnalysesFcn(Enum):
    PRINT_NUMBER_OF_ITERATIONS = CustomAnalysesImplementation.print_number_of_iterations


class CustomAnalyses:
    def __init__(self, analyses: tuple[CustomAnalysesFcn, ...]):
        self.analyses: tuple[Union[Callable, CustomAnalysesFcn], ...] = analyses

    def perform(self, studies):
        for analysis in self.analyses:
            analysis(studies)
