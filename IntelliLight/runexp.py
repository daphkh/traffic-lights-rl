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
    # [["cross.2phases_rou01_equal_300s.xml"], ["cross.2phases_rou01_equal_300s.xml"]],
    # [["cross.2phases_rou01_unequal_5_300s.xml"], ["cross.2phases_rou01_unequal_5_300s.xml"]],
    # [["cross.all_synthetic.rou.xml"], ["cross.all_synthetic.rou.xml"]],
]

list_model_name = [
                   # "Deeplight"#g,
                   # "DeeplightEquity",
                   "Pressure"
                   ]



# ================================= only change these two ========================================


import random
random.seed(SEED)
import numpy as np
np.random.seed(SEED)


import tensorflow as tf
print("devices:{",tf.test.gpu_device_name(),"}")
print("--",tf.test.is_gpu_available(),"--")
print(tf.config.list_physical_devices())


print("mid installs")
tf.random.set_seed(SEED)
import json
import os
import traffic_light_dqn
import traffic_light_equity_dqn
import traffic_light_pressure
import time
import datetime
import tensorflow as tf
print("post installs")

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

PATH_TO_CONF = os.path.join("conf", setting_memo)


model_func_dict = {"Deeplight" : traffic_light_dqn,
                    "DeeplightEquity" : traffic_light_equity_dqn,
                    "Pressure" : traffic_light_pressure}

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


final_results_string = ""

curTime = datetime.datetime.now()

with open("final_results_{0}.txt".format(curTime.strftime('%Y.%m.%d_%Hh%Mm%Ss')), "w") as fp:
    fp.write("Results!\n")

print("post sumo stuf")
for model_name in list_model_name:
    final_results_string += model_name + ":\n"
    for traffic_file, traffic_file_pretrain in list_traffic_files:
        start_time = time.time()
        final_results_string += "\t" + "{0}".format(traffic_file) + "\t"
        dic_exp = json.load(open(os.path.join(PATH_TO_CONF, "exp.conf"), "r"))
        dic_exp["MODEL_NAME"] = model_name
        dic_exp["TRAFFIC_FILE"] = traffic_file
        dic_exp["TRAFFIC_FILE_PRETRAIN"] = traffic_file_pretrain
        if "real" in traffic_file[0]:
            dic_exp["RUN_COUNTS"] = 8640
        elif "2phase" in traffic_file[0]:
            dic_exp["RUN_COUNTS"] = 43200
        elif "synthetic" in traffic_file[0]:
            dic_exp["RUN_COUNTS"] = 21600
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

# sumoCmd_nogui, sumoCmd_nogui_pretrain sumoCmd_pretrain sumoCmd
#traffic_light_pressure   traffic_light_dqn
        travel_time_overall, travel_time_priority = model_func_dict[model_name].main(memo=setting_memo, f_prefix=prefix,
                                                                                    sumo_cmd_str=sumoCmd, sumo_cmd_str_gui=sumoCmd,
                                                                                    sumo_cmd_pretrain_str=sumoCmd_nogui_pretrain,
                                                                                    sumo_cmd_pretrain_str_gui=sumoCmd_nogui_pretrain)

        final_results_string += "Overall Travel Time: " + str(travel_time_overall) + "\t Priority Travel Time: " + str(travel_time_priority) + "\n"
        final_results_string += "\t\tTook " + str(np.round((time.time() - start_time) / 60,2)) + " minutes\n"
        with open("final_results_{0}.txt".format(curTime.strftime('%Y.%m.%d_%Hh%Mm%Ss')), "a") as fp:
            fp.write(final_results_string)
        final_results_string = ""
        print("finished {0}".format(traffic_file))
    print ("finished {0}".format(model_name))
    print(final_results_string)
