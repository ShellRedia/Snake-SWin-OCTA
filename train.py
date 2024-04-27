import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import DataParallel, L1Loss

import os
import time
import random
import numpy as np
from tqdm import tqdm
from options import *
from loss_functions import *
from dataset import DataLoader_Producer
from metrics import MetricsStatistics
from monai.networks.nets import *
from models.SwinSnake import DSCNet, SwinSnake_Alter, SwinSnake_Dual

parser = argparse.ArgumentParser(description='training arguments')


device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class ModifiedModel(nn.Module):
    def __init__(self, original_model):
        super(ModifiedModel, self).__init__()
        self.original_model = original_model
        self.new_layer = nn.Sigmoid()

    def forward(self, x):
        x = self.original_model(x)
        x = self.new_layer(x)
        return x

class TrainManager:
    def __init__(self, model_dct, dataloader_producer):
        time_str = "-".join(["{:0>2}".format(x) for x in time.localtime(time.time())][:-3])
        config_str = "_".join(map(str, [*model_dct.values(), args.fov, args.label_type, args.epochs, args.remark]))
        self.record_dir = "/".join(["results", time_str, config_str])

        self.cpt_dir = "{}/checkpoints".format(self.record_dir)
        if not os.path.exists(self.cpt_dir): os.makedirs(self.cpt_dir)
        
        self.model = self.get_model(model_dct)

        self.dataloader_producer = dataloader_producer

        self.save_weight = model_dct["save_weight"]
        if self.save_weight: torch.save(self.model.state_dict(), '{}/init.pth'.format(self.cpt_dir))

        self.loss_func = DiceLoss()
        if args.label_type in ["Artery", "Vein", "LargeVessel"]:
            self.loss_func = lambda x, y: 0.8 * DiceLoss()(x, y) + 0.2 * clDiceLoss()(x, y)
        self.inputs_process = lambda x:x.to(torch.float).to(device)
    
    def get_model(self, model_dct):
        if "SwinSnake_Alter" == model_dct["name"]:
            model = SwinSnake_Alter(img_ch=3, output_ch=1, layer_depth=model_dct["layer_depth"], 
                            kernel_size=model_dct["kernel_size"], extend_scope=model_dct["extend_scope"], down_layer=model_dct["down_layer"],
                            rate=model_dct["rate"], repeat_n=model_dct["repeat_n"], device_id=args.device).to(device)
            
        elif "SwinSnake_Dual" == model_dct["name"]:
            model = SwinSnake_Dual(img_ch=3, output_ch=1, layer_depth=model_dct["layer_depth"], 
                            kernel_size=model_dct["kernel_size"], extend_scope=model_dct["extend_scope"],
                            rate=model_dct["rate"], device_id=args.device).to(device)

        elif "DSCNet" == model_dct["name"]:
            model = DSCNet(img_ch=3, output_ch=1, layer_depth=model_dct["layer_depth"], 
                           kernel_size=model_dct["kernel_size"], extend_scope=model_dct["extend_scope"],
                            rate=model_dct["rate"], device_id=args.device).to(device)

        elif model_dct["name"] == "SwinUNETR":
            model = SwinUNETR(img_size=(512,512), in_channels=3, out_channels=1, feature_size=24*model_dct["layer_depth"], spatial_dims=2)
            model = ModifiedModel(model).to(device)

        elif model_dct["name"] == "SwinUNETR":
            model = SwinUNETR(img_size=(512,512), in_channels=3, out_channels=1, feature_size=12*model_dct["layer_depth"], spatial_dims=2)
            model = ModifiedModel(model).to(device)

        elif model_dct["name"] == "UNet":
            N, B = 5, 8
            channels = [2 ** x for x in range(B, B+N)]
            strides = [2] * (len(channels) - 1)
            model = UNet(in_channels=3, out_channels=1, spatial_dims=2, channels=channels,strides=strides)
            model = ModifiedModel(model).to(device)
        
        elif model_dct["name"] == "SegResNet":
            model = SegResNet(in_channels=3, out_channels=1, spatial_dims=2)
            model = ModifiedModel(model).to(device)

        elif model_dct["name"] == "FlexUNet":
            model = FlexUNet(in_channels=3, out_channels=1, spatial_dims=2, backbone="efficientnet-b4")
            model = ModifiedModel(model).to(device)

        elif model_dct["name"] == "DiNTS":
            dints_space = TopologyInstance(spatial_dims=2, device="cuda:{}".format(args.device))
            model = DiNTS(dints_space=dints_space, in_channels=3, num_classes=1, spatial_dims=2)
            model = ModifiedModel(model).to(device)


        return model
    
    def reset(self):
        if self.save_weight: self.model.load_state_dict(torch.load('{}/init.pth'.format(self.cpt_dir)))
        pg = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(pg, lr=1, weight_decay=1e-4)
        epoch_p = args.epochs // 5
        lr_lambda = lambda x: max(1e-4, args.lr * x / epoch_p if x <= epoch_p else args.lr * 0.97 ** (x - epoch_p))
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train(self):
        train_loader, val_loader, test_loader = self.dataloader_producer.get_data_loader_ipn_v2()
        self.reset()
        metrics_statistics = MetricsStatistics(save_dir=self.record_dir)
        self.record_performance(train_loader, val_loader, test_loader, 0, metrics_statistics) # untrained performance
        # training loop:
        for epoch in tqdm(range(1, args.epochs+1), desc="training"):
            for samples, labels, sample_ids in train_loader:
                samples, labels = map(self.inputs_process, (samples, labels))
                self.optimizer.zero_grad()
                preds = self.model(samples)
                self.loss_func(preds, labels).backward()
                self.optimizer.step()
            self.scheduler.step()
            if epoch % (args.epochs // 10) == 0:
                self.record_performance(train_loader, val_loader, test_loader, epoch, metrics_statistics)
    
    def record_performance(self, train_loader, val_loader, test_loader, epoch, metrics_statistics):
        save_dir = "{}/{:0>4}".format(self.record_dir, epoch)
        if self.save_weight: torch.save(self.model.state_dict(), '{}/{:0>4}.pth'.format(self.cpt_dir, epoch))

        metrics_statistics.metric_values["learning rate"].append(self.optimizer.param_groups[0]['lr'])

        def record_dataloader(dataloader, loader_type="val", is_complete=True):
            with torch.no_grad():
                for images, labels, sample_ids in dataloader:
                    images, labels = map(self.inputs_process, (images, labels))
                    preds = self.model(images)
                    metrics_statistics.metric_values["loss_"+loader_type].append(self.loss_func(preds, labels).cpu().item())

                    preds = torch.gt(preds, 0.8).int()
                    sample_id = str(sample_ids[0])

                    to_cpu = lambda x:x[0][0].cpu().detach().int()
                    label, pred = to_cpu(labels), to_cpu(preds)
                    metrics_statistics.cal_epoch_metric(args.metrics, "{}-{}".format(args.label_type, loader_type), label, pred)
                    
                    if is_complete:
                        image, label, pred  = to_cpu(images * 255), label * 255, pred * 255
                        if not os.path.exists(save_dir): os.makedirs(save_dir)
                        save_sample_func = lambda x, y: np.save("/".join([save_dir,"{}_{}_{}.npy".format(args.label_type, x, sample_id)]), y)
                        for x, y in ("sample", image), ("label", label), ("pred", pred): save_sample_func(x, y)
    
        record_dataloader(train_loader, "train", False)
        record_dataloader(val_loader, "val", epoch == args.epochs)
        if test_loader: record_dataloader(test_loader, "test", epoch == args.epochs)

        metrics_statistics.record_result(epoch)

if __name__=="__main__":
    model_dct = {
        "name": args.model_name, 
        "layer_depth":args.layer_depth, 
        "kernel_size" : args.kernel_size,
        "extend_scope" : args.extend_scope,
        "rate" : args.rate,
        "down_layer" : args.down_layer,
        "repeat_n" : args.repeat_n,
        "save_weight":args.save_weight
    }
    is_resize = True
    if "SwinSnake" in model_dct["name"] or "DSCNet" in model_dct["name"]: is_resize = False
    dataloader_producer = DataLoader_Producer(fov=args.fov, label_type=args.label_type, batch_size=args.batch_size, is_resize=is_resize)
    train_manager = TrainManager(model_dct=model_dct, dataloader_producer=dataloader_producer)
    train_manager.train()