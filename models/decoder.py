'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui@ieee.org
'''

import torch
from torch import nn
from mmcls.SAVSS_dev.models.SAVSS.SAVSS import SAVSS
from models.MFS import MFS

class Decoder(nn.Module):
    def __init__(self, backbone, args=None):
        super().__init__()
        self.args = args
        self.backbone = backbone
        self.MFS = MFS(8)

    def forward(self, samples):
        outs_SAVSS = self.backbone(samples)
        out = self.MFS(outs_SAVSS)

        return out

def build(args):
    device = torch.device(args.device)
    args.device = torch.device(args.device)

    backbone = SAVSS(arch='Crack',
                     out_indices=(0, 1, 2, 3),
                     drop_path_rate=0.2,
                     final_norm=True,
                     convert_syncbn=True)
    model = Decoder(backbone, args)
    weights = torch.tensor(args.class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    criterion.to(device)

    return model, criterion