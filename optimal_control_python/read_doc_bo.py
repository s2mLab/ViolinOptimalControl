import time

from biorbd_optim import OptimalControlProgram

file_path = "/home/theophile/Documents/programmation/ViolinOptimalControl/optimal_control_python/results/2020_6_2_upDown_6c_m_f_if.bo"

if not isinstance(file_path, str):
    t = time.localtime(time.time())
    file_path = f"results/{t.tm_year}_{t.tm_mon}_{t.tm_mday}_upDown.bo"

OptimalControlProgram.read_information(file_path)
