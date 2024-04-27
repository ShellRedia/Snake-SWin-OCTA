import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import time

alpha = 0.5

overlay = lambda x, y: cv2.addWeighted(x, alpha, y, 1-alpha, 0)

# 灰度图像->单通道图像, Grayscale image -> single-channel image
to_blue = lambda x: np.array([x, np.zeros_like(x), np.zeros_like(x)]).transpose((1,2,0)).astype(dtype=np.uint8)
to_red = lambda x: np.array([np.zeros_like(x), np.zeros_like(x), x]).transpose((1,2,0)).astype(dtype=np.uint8)
to_green = lambda x: np.array([np.zeros_like(x), x, np.zeros_like(x)]).transpose((1,2,0)).astype(dtype=np.uint8)
to_light_green = lambda x: np.array([np.zeros_like(x), x / 2, np.zeros_like(x)]).transpose((1,2,0)).astype(dtype=np.uint8)
to_yellow = lambda x: np.array([np.zeros_like(x), x, x]).transpose((1,2,0)).astype(dtype=np.uint8)

to_3ch = lambda x: np.array([x,x,x]).transpose((1,2,0)).astype(dtype=np.uint8)

def show_result_sample_figure(image, label, pred):
    cvt_img = lambda x: x.astype(np.uint8)
    image, label, pred = map(cvt_img, (image, label, pred))
    if len(image.shape) == 2: image = to_3ch(image)
    else: image = image.transpose((1, 2, 0))
    label, pred = cv2.resize(label, image.shape[:2]), cv2.resize(pred, image.shape[:2])
    label_img = overlay(image, to_green(label))
    pred_img = overlay(image, to_yellow(pred))

    return np.concatenate((image, label_img, pred_img), axis=1)

def show_prompt_points_image(image, positive_region, negative_region, positive_points, negative_points, save_file=None):
    overlay_img = overlay(to_red(negative_region), to_yellow(positive_region))
    overlay_img = overlay(to_3ch(image), overlay_img)

    for x, y in positive_points: cv2.circle(overlay_img, (x, y), 4, (0, 255, 0), -1)
    for x, y in negative_points: cv2.circle(overlay_img, (x, y), 4, (0, 0, 255), -1)

    if save_file: cv2.imwrite(save_file, overlay_img)

    return overlay_img

def view_result_samples(result_dir):
    save_dir = "sample_display/{}".format(result_dir[len("results/"):])
    print("save_dir:")
    print(save_dir)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    sample_files = sorted(os.listdir(result_dir))
    p = sample_files[0].rfind("_") + 1
    file_names = [x[p:-4] for x in sample_files if "label" in x]
    data_name = [x[:p-7] for x in sample_files if "label" in x][0]
    for file_name in tqdm(file_names):
        label = np.load("{}/{}_label_{}.npy".format(result_dir, data_name, file_name))
        pred = np.load("{}/{}_pred_{}.npy".format(result_dir, data_name, file_name))
        image = np.load("{}/{}_sample_{}.npy".format(result_dir, data_name, file_name))

        result = show_result_sample_figure(image, label, pred)
        cv2.imwrite("{}/{}.png".format(save_dir, file_name), result)

def display_plot(conditions=["A", "B", "C", "D", "E"],
                g_label={"Dice-3M":[0.1, 0.2, 0.3, 0.4, 0.5], "Dice-6M":[0.3, 0.9, 0.98, 0.99, 0.92]}, 
                label_type = "RV",
                xlabel="X-axis Label",
                ylabel="Y-axis Label",
                ylim=(0.7, 1),
                title="default",
                save_file="default_plot.png"):
    label2color = {
        "RV":"green",
        "FAZ":"black",
        "Capillary":"yellow",
        "Artery":"red",
        "Vein":"blue"
    }
    metric2color = {
        "Dice": "green",
        "Jaccard" : "blue",
        "Hausdorff" : "black"
    }
    label2marker = {
        "RV":"o",
        "FAZ":"s",
        "Capillary":"D",
        "Artery":"+",
        "Vein":"x"
    }
    fov2style = {"3M":"-", "6M":"--"}

    plt.figure(figsize=(5, 5))

    
    

    for label_info, values in g_label.items():
        metric_type, fov = label_info.split("-")
        va = 'bottom' if metric_type == "Dice" else "top"
        plt.plot(conditions, values, label=label_info, linestyle=fov2style[fov],
                 marker=label2marker[label_type], markersize=5, color=metric2color[metric_type])
        for x, y in enumerate(values): 
            if va == "top": plt.text(x, y+0.04, str(y), ha='center', va=va, fontsize=10)
            else: plt.text(x, y, str(y), ha='center', va=va, fontsize=10)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # plt.ylim(*ylim)

    # plt.yticks([])

    plt.box(True)
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
    plt.tight_layout()

    # plt.legend()
    plt.legend().set_visible(False)
    plt.savefig(save_file)

def sample_comparison(sample_dirs):
    time_str = "-".join(["{:0>2}".format(x) for x in time.localtime(time.time())][:-3])
    save_dir = "sample_display/{}".format(time_str)
    print("save_dir:")
    print(save_dir)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    sample_merge = defaultdict(list)
    for sample_name in tqdm(os.listdir(sample_dirs[0]), desc="collecting..."):
        for sample_dir in sample_dirs:
            image = cv2.imread("{}/{}".format(sample_dir, sample_name), cv2.IMREAD_COLOR)
            sample_merge[sample_name].append(image)
    for file_name, sample_lst in tqdm(sample_merge.items(), desc="writing..."):
        image_merge = np.concatenate(sample_lst, axis=1)
        cv2.imwrite("{}/{}".format(save_dir, file_name), image_merge)



if __name__=="__main__":
    # display
    result_dir = r"results\2024-04-27-19-55-46\SwinSnake_Alter_3_11_1_72_MaxPooling_1_True_3M_FAZ_100_#\0100"
    view_result_samples(result_dir)

    # contrast

    # sample_dirs = [
    #     "sample_display/2024-03-30-20-49-53/SwinSnake_3M_LargeVessel_50/0010",
    #     "sample_display/2024-03-30-20-49-53/SwinSnake_3M_LargeVessel_50/0040"
    # ]
    # sample_comparison(sample_dirs)