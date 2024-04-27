import argparse

parser = argparse.ArgumentParser(description='training argument values')

# training:
parser.add_argument("-device", type=str, default="0")
parser.add_argument("-seed", type=int, default=42)
parser.add_argument("-epochs", type=int, default=100)
parser.add_argument("-batch_size", type=int, default=1)
parser.add_argument("-lr", type=float, default=0.001)
parser.add_argument("-k_fold", type=int, default=10)
parser.add_argument("-save_weight", type=bool, default=True)

# dataset:
parser.add_argument("-fov", type=str, default="3M")
parser.add_argument("-label_type", type=str, default="FAZ") # "LargeVessel", "FAZ", "Artery", "Vein", "Capillary"  

# deep model:
parser.add_argument("-model_name", type=str, default="SwinSnake_Alter")
parser.add_argument("-layer_depth", type=int, default=3)
parser.add_argument("-kernel_size", type=int, default=11)
parser.add_argument("-extend_scope", type=int, default=1)
parser.add_argument("-down_layer", type=str, default="MaxPooling")
parser.add_argument("-rate", type=int, default=72)
parser.add_argument("-repeat_n", type=int, default=1)


# evaluation
parser.add_argument("-metrics", type=list, default=["Dice", "Jaccard", "Hausdorff"])

# others:
parser.add_argument("-remark", type=str, default="#")

args = parser.parse_args()