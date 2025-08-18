import os.path
import cv2
from PIL import Image
from .base_dataset import BaseDataset
from .image_folder import make_dataset
from .utils import MaskToTensor
import torch

import albumentations as A  
from albumentations.pytorch import ToTensorV2  

class CrackDataset(BaseDataset):
    """A dataset class for crack dataset."""

    def __init__(self, args):
        BaseDataset.__init__(self, args)
        self.phase = args.phase  # "train", "val", or "test"
        self.img_paths = make_dataset(os.path.join(args.dataset_path, f'{self.phase}_img'))
        self.lab_dir   = os.path.join(args.dataset_path, f'{self.phase}_lab')
        

        # train 시에만 적용할 augmentation
        if self.phase == 'train':
            # 공통 증강: Resize
            self.common_transform = A.Compose(
                [
                    A.Resize(args.load_height, args.load_width, interpolation=cv2.INTER_LINEAR),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomGridShuffle(p=0.5, grid=(3, 3))
                ],
                additional_targets={'mask': 'mask'}
            )
            # train 전용 image, mask 변환
            self.img_transform = A.Compose([
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
                A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5), max_pixel_value=255.0),
                ToTensorV2()
            ])
            self.mask_transform = A.Compose(
                [ ToTensorV2() ],
                additional_targets={'mask': 'mask'}
            )
        else:
            # val/test 시엔 단순 Resize + Tensor/Normalize만
            self.resize_only = A.Compose([
                A.Resize(args.load_height, args.load_width, interpolation=cv2.INTER_LINEAR),
            ], additional_targets={'mask': 'mask'})

            self.img_transform = A.Compose([
                A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5), max_pixel_value=255.0),
                ToTensorV2()
            ])
            # mask는 ToTensorV2 없이 직접 Tensor 변환nsorV2와 동일하지만 추가_targets가 없어서 Ke
            # (mask 값이 0/1 이므로 ToTeyError 방지)
        
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        lab_path = os.path.join(self.lab_dir,
                                os.path.splitext(os.path.basename(img_path))[0] + '.png')

        # --- 1. 로드 ---
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lab = cv2.imread(lab_path, cv2.IMREAD_UNCHANGED)
        if lab.ndim == 3:
            lab = cv2.cvtColor(lab, cv2.COLOR_BGR2GRAY)
        _, lab = cv2.threshold(lab, 127, 1, cv2.THRESH_BINARY)

        # --- 2. phase별 분기 처리 ---
        if self.phase == 'train':
            # 공통 증강 + resize
            aug = self.common_transform(image=img, mask=lab)
            img_aug, lab_aug = aug['image'], aug['mask']
            # train 전용 image, mask 변환
            img_tensor = self.img_transform(image=img_aug)['image']
            m = self.mask_transform(image=lab_aug, mask=lab_aug)
            mask_tensor = m['mask'].unsqueeze(0).float()

        else:
            # val/test: 단순 resize만
            aug = self.resize_only(image=img, mask=lab)
            img_rs, lab_rs = aug['image'], aug['mask']
            # normalization & to-tensor
            img_tensor  = self.img_transform(image=img_rs)['image']
            # mask_transform 대신 직접 tensor 변환
            mask_tensor = torch.from_numpy(lab_rs).unsqueeze(0).float()

        return {
            'image':   img_tensor,    # [3, H, W]
            'label':   mask_tensor,   # [1, H, W]
            'A_paths': img_path,
            'B_paths': lab_path
        }

    def __len__(self):
        return len(self.img_paths)
