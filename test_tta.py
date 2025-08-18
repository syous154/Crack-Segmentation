'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui@ieee.org
'''

import numpy as np
import torch
import argparse
import os
import cv2

import ttach as tta  # <-- 추가
from datasets import create_dataset
from models import build_model
from main import get_args_parser

parser = argparse.ArgumentParser('SCSEGAMBA FOR CRACK', parents=[get_args_parser()])
args = parser.parse_args()
args.phase = 'val'
args.dataset_path = './data'

if __name__ == '__main__':
    args.batch_size = 1
    t_all = []
    device = torch.device(args.device)
    test_dl = create_dataset(args)
    load_model_file = "./checkpoints/weights/checkpoint_best.pth"
    data_size = len(test_dl)
    model, criterion = build_model(args)
    state_dict = torch.load(load_model_file, map_location=device)
    model.load_state_dict(state_dict["model"])
    model.to(device).eval()
    print("Load Model Successful!")

    # ------------------------------
    # TTA 설정
    # 1) d4: 0/90/180/270 회전 + H/V 플립 조합
    d4 = tta.aliases.d4_transform()
    transforms = d4

    # 모델 래핑: 평균 병합(merge_mode='mean')
    tta_model = tta.SegmentationTTAWrapper(model, transforms, merge_mode='mean')  # :contentReference[oaicite:4]{index=4}
    # ------------------------------

    suffix = load_model_file.split('/')[-2]
    save_root = "./results/results_test/TTA_d4_CLAHE_{}".format(suffix)
    if not os.path.isdir(save_root):
        os.makedirs(save_root)

    with torch.no_grad():
        for batch_idx, (data) in enumerate(test_dl):
            x = data["image"].to(device)          # [B,C,H,W]
            target = data["label"].to(device)

            # --- TTA 추론 ---
            out = tta_model(x)                    # [B,Classes,H,W] 로짓/확률(모델 출력 형식 유지)
            # ----------------

            # 멀티클래스 argmax → 단일 채널 마스크
            out = torch.argmax(out, dim=1).squeeze(0)   # [H,W]
            out = (out.detach().cpu().numpy().astype(np.uint8)) * 255

            # 저장
            root_name = data["A_paths"][0].split("/")[-1][0:-4]
            # target 시각화용(선택)
            tgt = target[0, 0, ...].detach().cpu().numpy()
            if np.max(tgt) > 0:
                tgt = (255 * (tgt / np.max(tgt))).astype(np.uint8)
            else:
                tgt = tgt.astype(np.uint8)

            print('----------------------------------------------------------------------------------------------')
            print(os.path.join(save_root, "{}_pre.png".format(root_name)))
            print('----------------------------------------------------------------------------------------------')
            cv2.imwrite(os.path.join(save_root, "{}_pre.png".format(root_name)), out)
            cv2.imwrite(os.path.join(save_root, "{}_lab.png".format(root_name)), tgt)

    print("Finished!")

