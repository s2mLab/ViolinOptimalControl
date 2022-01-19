from typing import Callable, Union
from enum import Enum
from statistics import mean

import numpy as np
from bioptim import Solution

from .enums import DataType


class LatexAnalysesFunctionImplementation:
    @staticmethod
    def mean_number_iterations(study_index: int, studies, solution: Solution, all_iterations: tuple[Solution, ...]) -> tuple[str, str]:
        header = r"Nombre\\d'itérations" if len(all_iterations) == 1 else r"Nombre moyen\\d'itérations"
        values = mean([sol.iterations for sol in all_iterations])
        return header, f"${values:0.0f}$" if len(all_iterations) == 1 else f"${values:0.1f}$"

    @staticmethod
    def mean_optimisation_time(study_index: int, studies, solution: Solution, all_iterations: tuple[Solution, ...]) -> tuple[str, str]:
        header = r"Temps\\d'optimisation\\(s)" if len(all_iterations) == 1 else r"Temps moyen\\d'optimisation\\(s)"
        return header, f"${mean([sol.real_time_to_optimize for sol in all_iterations]):0.1f}$"

    @staticmethod
    def total_optimisation_time(study_index: int, studies, solution: Solution, all_iterations: tuple[Solution, ...]) -> tuple[str, str]:
        header = r"Temps total\\d'optimisation\\(s)"
        return header, f"${sum([sol.real_time_to_optimize for sol in all_iterations]):0.1f}$"

    @staticmethod
    def mean_iteration_time(study_index: int, studies, solution: Solution, all_iterations: tuple[Solution, ...]) -> tuple[str, str]:
        n_iter = sum([sol.iterations for sol in all_iterations])
        total_time = sum([sol.real_time_to_optimize for sol in all_iterations])
        header = r"Temps moyen\\par itération\\(s/itération)"
        return header, f"${total_time / n_iter:0.3f}$"

    @staticmethod
    def rmse_q(study_index: int, studies, solution: Solution, all_iterations: tuple[Solution, ...]) -> tuple[str, str]:
        reference_index = studies.studies[study_index].rmse_index

        data_ref = getattr(studies.solutions[reference_index][0], DataType.STATES.value)["q"]
        data = getattr(solution, DataType.STATES.value)["q"]

        e = data_ref - data
        se = e ** 2
        mse = np.sum(se, axis=1) / data_ref.shape[1]
        rmse = np.sqrt(mse)
        sum_rmse = np.sum(rmse)

        if sum_rmse == 0:
            rmse_str = "---"
        else:
            rmse_str = f"{sum_rmse:0.3e}"
            if rmse_str.find("e") >= 0:
                rmse_str = rmse_str.replace("e", "\\times 10^{{")
                rmse_str += "}}"
                rmse_str = rmse_str.replace("+0", "")
                rmse_str = rmse_str.replace("-0", "-")
                rmse_str = rmse_str.replace("\\times 10^{{0}}", "")
            rmse_str = "$" + rmse_str + "$"

        header = r"$\sum\text{\rmse{}}$\\pour $\q$\\(rad)"
        return header, rmse_str


class LatexAnalysesFcn(Enum):
    MEAN_NUMBER_ITERATIONS = LatexAnalysesFunctionImplementation.mean_number_iterations
    TOTAL_OPTIMIZATION_TIME = LatexAnalysesFunctionImplementation.total_optimisation_time
    MEAN_OPTIMIZATION_TIME = LatexAnalysesFunctionImplementation.mean_optimisation_time
    MEAN_ITERATION_TIME = LatexAnalysesFunctionImplementation.mean_iteration_time
    RMSE_Q = LatexAnalysesFunctionImplementation.rmse_q


class LatexTable:
    def __init__(
            self,
            table_caption: str,
            table_label: str,
            analyses: tuple[LatexAnalysesFcn, ...],
            add_non_converged_notice: bool,
            add_bfgs_dagger_notice: bool
    ):
        self.table_caption = table_caption
        self.add_non_converged_notice = add_non_converged_notice
        self.non_converged_text = "Condition n'ayant pas convergé (maximum d'itérations atteint)"
        self.add_bfgs_dagger_to_caption = add_bfgs_dagger_notice
        self.table_caption += f"$\\dagger$"
        self.bfgs_dagger_text = f"Le nombre d'itération et les temps rapportés n'incluent pas la " \
                                f"préoptimisation de $100$ itérations en utilisant l'approximation BFGS"
        self.table_label = table_label
        self.analyses: tuple[Union[Callable, LatexAnalysesFcn], ...] = analyses
        self.has_non_converging = False

        self.preamble = (
            f"\\documentclass{{article}}\n"
            f"\n"
            f"\\usepackage{{amsmath}}\n"
            f"\\usepackage{{amssymb}}\n"
            f"\\usepackage[table]{{xcolor}}\n"
            f"\\usepackage{{threeparttable}}\n"
            f"\\usepackage{{makecell}}\n"
            f"\\definecolor{{lightgray}}{{gray}}{{0.91}}\n"
            f"\n\n"
            f"% Aliases\n"
            f"\\newcommand{{\\rmse}}{{RMSE}}\n"
            f"\\newcommand{{\\ocp}}{{OCP}}\n"
            f"\\newcommand{{\\cyclicNMPC}}{{NMPC cyclique}}\n"
            f"\\newcommand{{\\multiCyclicNMPC}}{{NMPC multicyclique}}\n"
            f"\\newcommand{{\\controls}}{{\\mathbf{{u}}}}\n"
            f"\\newcommand{{\\states}}{{\\mathbf{{x}}}}\n"
            f"\\newcommand{{\\statesDot}}{{\\mathbf{{\\dot{{x}}}}}}\n"
            f"\\newcommand{{\\q}}{{\\mathbf{{q}}}}\n"
            f"\\newcommand{{\\qdot}}{{\\mathbf{{\\dot{{q}}}}}}\n"
            f"\\newcommand{{\\qddot}}{{\\mathbf{{\\ddot{{q}}}}}}\n"
            f"\\newcommand{{\\f}}{{\\mathbf{{f}}}}\n"
            f"\\newcommand{{\\taupm}}{{\\tau^{{\\pm}}}}\n"
            f"\\newcommand{{\\tauns}}{{\\tau^{{\\times}}}}\n"
            f"\n"
            f"\\newcommand{{\\condition}}{{C/}}\n"
            f"\\newcommand{{\\noFatigue}}{{\\varnothing}}\n"
            f"\\newcommand{{\\qcc}}{{4\\textsubscript{{CC}}}}\n"
            f"\\newcommand{{\\pe}}{{P\\textsubscript{{E}}}}\n"
            f"\\newcommand{{\\condTau}}{{{{\\condition}}{{\\tau}}{{}}}}\n"
            f"\\newcommand{{\\condTauNf}}{{{{\\condition}}{{\\tau}}{{\\noFatigue}}}}\n"
            f"\\newcommand{{\\condTauQcc}}{{{{\\condition}}{{\\tau}}{{\\qcc}}}}\n"
            f"\\newcommand{{\\condTauPe}}{{{{\\condition}}{{\\tau}}{{\\pe}}}}\n"
            f"\\newcommand{{\\condTaupm}}{{{{\\condition}}{{\\taupm}}{{}}}}\n"
            f"\\newcommand{{\\condTaupmQcc}}{{{{\\condition}}{{\\taupm}}{{\\qcc}}}}\n"
            f"\\newcommand{{\\condTaupmPe}}{{{{\\condition}}{{\\taupm}}{{\\pe}}}}\n"
            f"\\newcommand{{\\condTauns}}{{{{\\condition}}{{\\tauns}}{{}}}}\n"
            f"\\newcommand{{\\condTaunsQcc}}{{{{\\condition}}{{\\tauns}}{{\\qcc}}}}\n"
            f"\\newcommand{{\\condTaunsPe}}{{{{\\condition}}{{\\tauns}}{{\\pe}}}}\n"
            f"\\newcommand{{\\condAlpha}}{{{{\\condition}}{{\\alpha}}{{}}}}\n"
            f"\\newcommand{{\\condAlphaNf}}{{{{\\condition}}{{\\alpha}}{{\\noFatigue}}}}\n"
            f"\\newcommand{{\\condAlphaQcc}}{{{{\\condition}}{{\\alpha}}{{\\qcc}}}}\n"
            f"\\newcommand{{\\condAlphaPe}}{{{{\\condition}}{{\\alpha}}{{\\pe}}}}\n"
            f"\n\n"
        )
        self.begin_document = (
            f"\\begin{{document}}\n"
            f"\n"
        )
        self.begin_table = (
            f"\\begin{{table}}[!ht]\n"
            f" \\rowcolors{{1}}{{}}{{lightgray}}\n"
            f" \\caption{{{self.table_caption}}}\n"
            f" \\label{{{self.table_label}}}\n"
            f" \\begin{{threeparttable}}\n"
        )
        self.end_table = (
            f" \\end{{threeparttable}}\n"
            f"\\end{{table}}\n\n"
        )
        self.end_document = (
            f"\\end{{document}}\n"
        )

    def get_table_text(self, studies, all_solutions: list[tuple[Solution, list[Solution]], ...]) -> str:
        header = ""
        values = []
        for i, (solution, all_iterations) in enumerate(all_solutions):
            results = [analyse(i, studies, solution, all_iterations) for analyse in self.analyses]
            if i == 0:
                header = " & ".join([f"\\bfseries\\makecell[c]{{{result[0]}}}" for result in results])

            name_condition = studies.studies[i].name
            if self.add_non_converged_notice and True in [iteration.iterations == studies.studies[i].solver.max_iter for iteration in all_iterations]:
                name_condition += "*"
                self.has_non_converging = True
            values.append(name_condition + " & " + " & ".join([str(result[1]) for result in results]))
        all_values = " \\\\ \n   ".join(values)

        tabular = (
            f"  \\begin{{tabular}}{{l{'c' * len(self.analyses)}}}\n"
            f"   \\hline\n"
            f"   \\bfseries Condition & {header} \\\\ \n"
            f"   \\hline\n"
            f"   {all_values} \\\\ \n"
            f"   \\hline\n" 
            f"  \\end{{tabular}}\n"
        )

        if self.add_bfgs_dagger_to_caption or self.add_non_converged_notice:
            tabular += f"  \\begin{{tablenotes}}\n"
            if self.add_bfgs_dagger_to_caption:
                tabular += f"   \\item $\\dagger$ {self.bfgs_dagger_text}\n"
            if self.add_non_converged_notice and self.has_non_converging:
                tabular += f"   \\item * {self.non_converged_text}\n"
            tabular += f"  \\end{{tablenotes}}\n"

        return self.preamble + self.begin_document + self.begin_table + tabular + self.end_table + self.end_document
