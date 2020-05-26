import time

from biorbd_optim import OptimalControlProgram

file_path = "results/2020_5_26_upDown_6c_m_f_i.bo"

if not isinstance(file_path, str):
    t = time.localtime(time.time())
    file_path = f"results/{t.tm_year}_{t.tm_mon}_{t.tm_mday}_upDown.bo"

OptimalControlProgram.read_information(file_path)
