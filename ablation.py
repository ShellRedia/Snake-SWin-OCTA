from itertools import product
import time, GPUtil
from concurrent.futures import ThreadPoolExecutor
import os

available_devices = ",".join(map(str, range(len(GPUtil.getGPUs()))))

# "LargeVessel", "FAZ", "Artery", "Vein", "Capillary"          
option_dct = {
    "-fov" : ["6M"],
    "-label_type" : ["Artery", "Vein"],
    "-epochs" : [100],
    "-rate" : [48],
    "-model_name" : ["DSCNet"], # "SwinSnake_Alter", "SwinSnake_Dual", "DSCNet", "UNet", "SwinUNETR", "SegResNet", "FlexUNet", "DiNTS"
    "-repeat_n": [1],
    "-down_layer" : ["MaxPooling"],
    "-layer_depth" : [3],
    "-kernel_size" : [9], # little in influence, 7 is the best in [3, 7, 11]
    "-extend_scope" : [2],
    "-remark" : ["Large"]
}

task_lst = list(product(*option_dct.values()))

executor = ThreadPoolExecutor(max_workers=len(task_lst))

inner_interval, outer_interval = 60, 120

option_names = list(option_dct.keys())

while task_lst:
    for gpu_id, gpu in enumerate(GPUtil.getGPUs()):
        if task_lst:
            if str(gpu_id) in available_devices.split(",") and gpu.memoryFree / gpu.memoryTotal > 0.5:
                option_vals = task_lst.pop(0)
                task_start_cmd = "python train.py -device {} ".format(gpu_id)
                for name, val in zip(option_names, option_vals): task_start_cmd += "{} {} ".format(name, val)
                print(task_start_cmd)
                executor.submit(os.system, task_start_cmd)
                time.sleep(inner_interval)
    time.sleep(outer_interval)