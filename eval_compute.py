'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui@ieee.org
'''

from thop import profile
import torch
from main import get_args_parser
import argparse
from models.decoder import build

parser = argparse.ArgumentParser('SCSEGAMBA FOR CRACK', parents=[get_args_parser()])
args = parser.parse_args()

if __name__ == '__main__':
    model, _, = build(args)
    model.to(args.device)

    input = torch.randn(1, 3, 512, 512)
    samples = input.to(torch.device(args.device))

    flops, params = profile(model, (samples, ))
    print("flops(G):", flops/1e9, "params(M):", params/1e6)
