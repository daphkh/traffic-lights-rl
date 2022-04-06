# -*- coding: utf-8 -*-

'''
@author: hzw77, gjz5038

python runexp.py

Run experiments in batch with configuration

'''

# ================================= only change these two ========================================
SEED = 31200

setting_memo = "one_run"


# first column: for train, second column: for spre_train
list_traffic_files = [
    [["cross.2phases_rou1_switch_rou0.xml"], ["cross.2phases_rou1_switch_rou0.xml"]],
    [["cross.2phases_rou01_equal_300s.xml"], ["cross.2phases_rou01_equal_300s.xml"]],
    [["cross.2phases_rou01_unequal_5_300s.xml"], ["cross.2phases_rou01_unequal_5_300s.xml"]],
    [["cross.all_synthetic.rou.xml"], ["cross.all_synthetic.rou.xml"]],
]

list_model_name = [
                   # "Deeplight",
                   "Pressure",
                   ]

# ================================= only change these two ========================================


import random
random.seed(SEED)
import numpy as np
np.random.seed(SEED)

from tensorflow import set_random_seed

print("mid installs")
set_random_seed((SEED))
import json
import os
import traffic_light_dqn
import traffic_light_pressure
import time
print("post installs")

PATH_TO_CONF = os.path.join("conf", setting_memo)


print("post path")
sumoBinary = r"C:\Users\CSung\Desktop\CIS522Project\sumo-win64-0.32.0\sumo-0.32.0\bin\sumo-gui.exe"
sumoCmd = [sumoBinary,
           '-c',
           r'{0}\\data\\{1}\\cross.sumocfg'.format(os.path.split(os.path.realpath(__file__))[0], setting_memo)]
sumoCmd_pretrain = [sumoBinary,
                    '-c',
                    r'{0}/data/{1}/cross_pretrain.sumocfg'.format(
                        os.path.split(os.path.realpath(__file__))[0], setting_memo)]

sumoBinary_nogui = r"C:\Users\CSung\Desktop\CIS522Project\sumo-win64-0.32.0\sumo-0.32.0\bin\sumo.exe"
sumoCmd_nogui = [sumoBinary_nogui,
                 '-c',
                 r'{0}\\data\\{1}\\cross.sumocfg'.format(
                     os.path.split(os.path.realpath(__file__))[0], setting_memo)]
sumoCmd_nogui_pretrain = [sumoBinary_nogui,
                          '-c',
                          r'{0}/data/{1}/cross_pretrain.sumocfg'.format(
                              os.path.split(os.path.realpath(__file__))[0], setting_memo)]


print("post sumo stuf")
for model_name in list_model_name:
    for traffic_file, traffic_file_pretrain in list_traffic_files:
        dic_exp = json.load(open(os.path.join(PATH_TO_CONF, "exp.conf"), "r"))
        dic_exp["MODEL_NAME"] = model_name
        dic_exp["TRAFFIC_FILE"] = traffic_file
        dic_exp["TRAFFIC_FILE_PRETRAIN"] = traffic_file_pretrain
        if "real" in traffic_file[0]:
            dic_exp["RUN_COUNTS"] = 86400
        elif "2phase" in traffic_file[0]:
            dic_exp["RUN_COUNTS"] = 72000
        elif "synthetic" in traffic_file[0]:
            dic_exp["RUN_COUNTS"] = 216000
        json.dump(dic_exp, open(os.path.join(PATH_TO_CONF, "exp.conf"), "w"), indent=4)

        # change MIN_ACTION_TIME correspondingly

        dic_sumo = json.load(open(os.path.join(PATH_TO_CONF, "sumo_agent.conf"), "r"))
        if model_name == "Deeplight":
            dic_sumo["MIN_ACTION_TIME"] = 5
        else:
            dic_sumo["MIN_ACTION_TIME"] = 1
        json.dump(dic_sumo, open(os.path.join(PATH_TO_CONF, "sumo_agent.conf"), "w"), indent=4)

        prefix = "{0}_{1}_{2}_{3}".format(
            dic_exp["MODEL_NAME"],
            dic_exp["TRAFFIC_FILE"],
            dic_exp["TRAFFIC_FILE_PRETRAIN"],
            time.strftime('%m_%d_%H_%M_%S_', time.localtime(time.time())) + "seed_%d" % SEED
        )

# sumoCmd_nogui, sumoCmd_nogui_pretrain
        traffic_light_pressure.main(memo=setting_memo, f_prefix=prefix, sumo_cmd_str=sumoCmd, sumo_cmd_pretrain_str=sumoCmd_pretrain)

        print("finished {0}".format(traffic_file))
    print ("finished {0}".format(model_name))
