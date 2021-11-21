import argparse
import logging
import os

import torch
import torch.nn.functional as F
import torch.utils.data.distributed
from torch.nn.utils import clip_grad_norm_
import tqdm

from backbones import get_model
from dataset import Inha_Face
from utils.utils_amp import MaxClipGradScaler
from utils.utils_config import get_config
from utils.utils_logging import AverageMeter, init_logging

import torch.nn as nn

import math

class ArcModule(nn.Module):
    def __init__(self, in_features, out_features, s = 10, m = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_normal_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = torch.tensor(math.cos(math.pi - m))
        self.mm = torch.tensor(math.sin(math.pi - m) * m)

    def forward(self, inputs, labels):
        cos_th = F.linear(inputs, F.normalize(self.weight))
        cos_th = cos_th.clamp(-1, 1)
        sin_th = torch.sqrt(1.0 - torch.pow(cos_th, 2))
        cos_th_m = cos_th * self.cos_m - sin_th * self.sin_m
        # print(type(cos_th), type(self.th), type(cos_th_m), type(self.mm))
        cos_th_m = torch.where(cos_th > self.th, cos_th_m, cos_th - self.mm)

        cond_v = cos_th - self.th
        cond = cond_v <= 0
        cos_th_m[cond] = (cos_th - self.mm)[cond]

        if labels.dim() == 1:
            labels = labels.unsqueeze(-1)
        onehot = torch.zeros(cos_th.size()).cuda()
        labels = labels.type(torch.LongTensor).cuda()
        onehot.scatter_(1, labels, 1.0)
        outputs = onehot * cos_th_m + (1.0 - onehot) * cos_th
        outputs = outputs * self.s
        return outputs

def main(args):
    cfg = args.parse_args()

    gpu_device = 0
    torch.cuda.set_device(gpu_device)

    
    train_set = Inha_Face()
    train_loader = torch.utils.data.DataLoader(train_set, shuffle = True, batch_size = 64, num_workers = 4)
    backbone = get_model(cfg.network, dropout=0.0, fp16=True, num_features=512).to(gpu_device)
    arc = ArcModule(512, cfg.num_classes)
    


    backbone_pth = "./backbone.pth"
    backbone.load_state_dict(torch.load(backbone_pth, map_location=torch.device(gpu_device)))
    backbone = backbone.to(0)
    arc = arc.to(0)
    backbone.train()
    arc.train()
    

    opt_backbone = torch.optim.Adam(backbone.parameters(), lr=cfg.lr)
    opt_arc = torch.optim.Adam(arc.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

        
    

    scheduler_backbone = torch.optim.lr_scheduler.CosineAnnealingLR(opt_backbone, cfg.num_epoch)
    scheduler_arc = torch.optim.lr_scheduler.CosineAnnealingLR(opt_arc, cfg.num_epoch)


    loss_window = AverageMeter()
    start_epoch = 0
    global_step = 0
    for epoch in range(start_epoch, cfg.num_epoch):
        loss_window.reset()
        for step, (img, label) in tqdm.tqdm(enumerate(train_loader), desc = f"{epoch}/{cfg.num_epoch}, total iter : {len(train_loader)}"):
            global_step += 1
            img = img.to(0)
            label = label.to(0)
            features = F.normalize(backbone(img))
            logits = arc(features, label)
            loss = criterion(logits, label)
            
            loss.backward()
            
            if epoch >= 3:
                clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)
                opt_backbone.step()
                opt_backbone.zero_grad()
            opt_arc.step()
            opt_arc.zero_grad()
                        
            scheduler_backbone.step()
            scheduler_arc.step()
            loss_window.update(loss.item())
            if step % 1000 == 0:
                print(f"avgloss : {loss_window.avg}")
        torch.save(backbone.state_dict(), "./r100_transfer_" + epoch + ".pth")
        torch.save(arc.state_dict(), "./arc_module_" + epoch + ".pth")
        print(f"{epoch}/{cfg.num_epoch} : avg_loss = {loss_window.avg}")
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument("--batch_size", default = 64)
    parser.add_argument("--num_classes", default = 86876)
    parser.add_argument("--network", default = "r100")
    parser.add_argument("--num_epoch", default = 6)
    parser.add_argument("--lr", default = 3e-4)
    main(parser)
