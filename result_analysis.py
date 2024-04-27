import os
import pandas as pd
import numpy as np
import shutil

from tqdm import tqdm
from collections import defaultdict

from display import display_plot

class ResultAnalysis:
    def __init__(self, 
                 result_dir="results",
                 start_time_str="2024-01",
                 end_time_str="2024-12"):
        self.result_dir = result_dir
        self.start_time_str = start_time_str
        self.end_time_str = end_time_str
        self.valid_dirs = []
        self.get_valid_results()

    def get_valid_results(self, filter=""):
        valid_dirs = []
        for datetime_dir in sorted(os.listdir(self.result_dir)):
            if self.start_time_str <= datetime_dir <= self.end_time_str:
                for task_dir in sorted(os.listdir("{}/{}".format(self.result_dir, datetime_dir))):
                    if not filter or filter in task_dir:
                        valid_dirs.append("{}/{}/{}".format(self.result_dir, datetime_dir, task_dir))
        self.valid_dirs = valid_dirs
        return valid_dirs
    
    def check_failure_results(self):
        failure_results = []
        for valid_dir in self.valid_dirs:
            ms_file_path = "{}/metrics_statistics.xlsx".format(valid_dir)
            if os.path.exists(ms_file_path):
                df = pd.read_excel(ms_file_path)
                task_options = valid_dir.split("/")[-1]
                total_epochs = int(task_options.split("_")[2])
                trained_epochs = int(list(df["epoch"])[-1])
                if total_epochs != trained_epochs:
                    task_dir = valid_dir.split("/")[1]
                    failure_results.append([task_dir, total_epochs, trained_epochs])
            else:
                task_dir = valid_dir.split("/")[1]
                failure_results.append([task_dir])
        return failure_results

    def delete_failure_results(self):
        for failure_result in tqdm(self.check_failure_results(), desc="remove invalid"):
            shutil.rmtree("{}/{}".format(self.result_dir, failure_result[0]))
        self.get_valid_results()
    
    def get_test_result(self, timestamp=""):
        task_dir = "{}/{}".format(self.result_dir, timestamp)
        rnt = []
        if os.path.exists(task_dir):
            task_options = os.listdir(task_dir)[0]
            # print("task options :", task_options)
            df = pd.read_excel("{}/{}/metrics_statistics.xlsx".format(task_dir, task_options))
            val_max_index = df["loss_val"].idxmin()
            for k in df:
                if "test" in k: 
                    # print("{} : {}".format(k, df[k][val_max_index]))
                    if "-" in k:
                        metric, value = k.split()[0], round(df[k][val_max_index], 4)
                        rnt.append((metric, value))
        return rnt

'''

'''
if __name__=="__main__":
    ra = ResultAnalysis()
    g = defaultdict(list)

    # Artery, Vein, LargeVessel, FAZ, Capillary
    label_type = "FAZ"
    fov = "3M"

    for x in ra.get_valid_results():
        if fov in x and label_type in x:
            print(x)
            datetime_dir = x.split("/")[1]
            g["condition"].append(x.split("/")[-1])
            for k, v in ra.get_test_result(datetime_dir):
                g[k].append(v)

    df = pd.DataFrame(g)
    df_sorted = df.sort_values(by='condition')

    print(df_sorted)