import matplotlib.pyplot as plt 
import numpy as np
import os
import json

res_dir = "res/output/"
run_name = "your-own-run-name" # defined by --output_dir in your scripts
checkpoint_num = 20000
all_results = {}
all_runs_list = [r for r in os.listdir(res_dir + run_name) if "misc" not in r]
print(all_runs_list)
for single_run in all_runs_list:
    for w in os.listdir(res_dir + run_name + "/" + single_run):
        if w not in all_results.keys():
            all_results[w] = []
        full_name = single_run
        target_file = os.path.join(res_dir, run_name, single_run, w, 
                                   f"checkpoint-{checkpoint_num}", "trainer_state.json")
        if os.path.isfile(target_file):
            with open(target_file, "r") as f_d:
                dict = json.load(f_d)
            lm_loss = dict["log_history"][-1]["loss"]
            mt_loss = dict["log_history"][-1]["mt_loss"] if "mt_loss" in dict["log_history"][-1] else -1
            all_results[w].append((full_name, lm_loss, mt_loss))

for w, l in all_results.items():
    print(w)
    print("sort by lm_loss:")
    for name, lm_loss, mt_loss in sorted(l, key=lambda x: x[1], reverse=False):
        print(f"{name}\t{lm_loss}\t{mt_loss}")

# 一维依存关系曲线图
loss_type = ["LM"]
for i in range(1):
    for w, l in all_results.items():
        l_1 = sorted(l, key=lambda x: float(x[0].split("_")[0]), reverse=False)
        x = [float(data[0].split("_")[0]) for data in l_1]
        y = [data[i+1] for data in l_1]
        plt.plot(x, y, label=w)
        # for _x, _y in zip(x, y):
            # plt.text(_x, _y, (_x, _y))
    plt.xlabel("lr")
    plt.ylabel("loss")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.savefig(f"logs/pics/{run_name}_{loss_type[i]}_step_{checkpoint_num}.png")
    plt.clf()