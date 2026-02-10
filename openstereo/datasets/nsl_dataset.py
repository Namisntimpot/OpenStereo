import os.path as osp
from pathlib import Path
import numpy as np
import sys
import torch
import torch.nn.functional as F
from .dataset_template import build_transform_by_cfg

NSL_DATALOADER_PATH = osp.join(*Path(__file__).parts[:-5])  # xxx/DeepSL/zoo/OpenStereo/stereo/datasets/__init__.py
sys.path.append(NSL_DATALOADER_PATH)

from deepsl_data.dataloader.dataloader import SimplifiedStereoDataset # noqa
from deepsl_data.dataloader.file_fetcher import LocalFileFetcher # noqa


def depth2disparity(depth, intri, extri, other_extri = None):
    # depth: [B, H, W]
    # intri: [B, 3, 3]
    # extri: [B, 4, 4]
    B, H, W = depth.shape
    assert intri.shape == (B, 3, 3)
    assert extri.shape == (B, 4, 4)
    # get the focal length
    f = intri[..., 0, 0]
    # get the baseline
    if other_extri is None:
        other_extri = torch.zeros_like(extri, dtype=extri.dtype, device=extri.device)
    b = torch.norm(extri[..., :3, 3] - other_extri[..., :3, 3], dim=-1)
    # b = extri[:, 0, 3]
    # get the disparity
    shape = f.shape + (1,)  * (depth.ndim - f.ndim)
    disparity = f.view(shape) * b.view(shape) / depth
    return disparity

def rectify_images_simplified(
        images:torch.Tensor, align_intri:torch.Tensor, origin_intri:torch.Tensor, normalized_intri:bool=True
    ):
    '''
    Rectify images so as to align the projector's intrinsic to the camera's  
    This function only considers a simplifed case where the proj and the cam have the same resolution.  
    images: (B,(C),H,W). Must be batched. If images.ndim==3, it will be considered as missing C dim instead of B dim.  
    *_intri: (B, 3, 3), camera space -> pixel space.  
    '''
    b = images.shape[0]
    ori_pat_dim = images.ndim
    if ori_pat_dim == 3:
        images = images.unsqueeze(1)  # (B,1,H,W)  
    h, w = images.shape[-2:]
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    pix = torch.stack((x, y), dim=-1).to(torch.float32).to(images.device)
    if normalized_intri:
        n = torch.tensor([w, h], dtype=torch.float32, device=images.device)
        pix = pix / n  # range (0,1)
    homogeneous_coord = torch.concat((pix, torch.ones_like(pix[...,:1], dtype=torch.float32, device=pix.device)), dim=-1) # (h,w,3)
    mat = torch.matmul(origin_intri, torch.linalg.inv(align_intri)) # (B, 3, 3)
    proj_coord = torch.matmul(mat.view(b,1,1,3,3), homogeneous_coord.view(h,w,3,1)).squeeze(-1)[..., :2]  # (b, h, w, 2)  
    if not normalized_intri:
        n = torch.tensor([w, h], dtype=torch.float32, device=images.device)
        proj_coord = proj_coord / n
    rectified = F.grid_sample(
        images, 2*proj_coord-1, mode='bilinear', padding_mode='zeros', align_corners=False
    )  # (b,c,h,w)
    return rectified.squeeze(1) if ori_pat_dim == 3 else rectified


class NSLDataset(SimplifiedStereoDataset):
    def __init__(self, data_info, data_cfg, mode):
        if mode.lower() == 'training':
            split = 'train'
        elif mode.lower() == 'evaluating':
            split = 'val'
        elif mode.lower() == 'testing':
            split = 'test'
        else:
            raise ValueError(f"Unknown mode: {mode}")
        data_root = data_info.DATA_SPLIT[mode.upper()]
        filefetcher = LocalFileFetcher(split, data_root, decomp=False, cleaned=True)

        self.load_type = data_info.get('LOAD_TYPE', 'left_right')  # left_right or left_patt
        assert self.load_type in ['left_right', 'left_patt'], "LOAD_TYPE must be 'left_right' or 'left_patt'"

        gray = data_info.get('GRAY', True)
        parameters = True
        patternname = None
        normal = False
        materialtype = False
        super().__init__(filefetcher, gray=gray, parameters=parameters,
                         patternname=patternname, normal=normal,
                         materialtype=materialtype)
        
        transform_config = data_cfg.DATA_TRANSFORM[mode.upper()]
        self.transform = build_transform_by_cfg(transform_config)

        # load patterns
        # load patterns.
        self.patterns_images = {
            k: torch.from_numpy(self.file_fetcher.fetch_pattern(k)) for k in self.list_patterns
        }  # 0-1, (h,w,3)


    def __getitem__(self, index):
        # 父类自带transform，干脆重新写.
        pattern_to_fetch = self.list_patterns[index % self.num_patterns]
        flatten_key_to_fetch = self.file_fetcher.flatten_keys()[index // self.num_patterns]
        data = self.file_fetcher.fetch(
            flatten_key_to_fetch, self.parameters, pattern_to_fetch, self.normal, self.materialtype)  # 图片已是(0,1)
        # data = to_tensor(data)   # 在transform里做
        if self.gray:
            data = self.convert_imgs_to_gray(data)
        # rename keys.
        for k in list(data.keys()):
            newk = k.split(".")[0] # if self.patternname is None else k 去掉文件后缀...
            prefix = newk[:2]
            if prefix != 'L_' and prefix != 'R_' and prefix != 'P_':
                newk = "_".join(newk.split("_")[1:])
            v = data.pop(k)
            data[newk] = v

        l_image, r_image = data['L_Image'], data['R_Image']
        patt = self.patterns_images[pattern_to_fetch]
        dep = data['L_Depth']
        l_intri, r_intri, p_intri = data['L_intri'], data['R_intri'], data['P_intri']
        l_extri, r_extri, p_extri = data['L_extri'], data['R_extri'], data['P_extri']

        if self.load_type == 'left_patt':
            # rectify.
            patt = rectify_images_simplified(
                torch.from_numpy(patt).unsqueeze(0).permute(0, 3, 1, 2), torch.from_numpy(l_intri), 
                torch.from_numpy(p_intri), False
            ).permute(0, 2, 3, 1).squeeze_(0).numpy()  # HWC
            p_intri = l_intri
            r_image = patt
            # r_intri = p_intri
            r_intri = l_intri
            r_extri = p_extri
        # dep2disp
        disp = depth2disparity(torch.from_numpy(dep).unsqueeze(0), torch.from_numpy(l_intri).unsqueeze(0), 
                               torch.from_numpy(l_extri).unsqueeze(0), torch.from_numpy(r_extri).unsqueeze(0)).squeeze_(0).numpy()
        assert l_image.shape == r_image.shape, f"Shape mismatch: {l_image.shape}, {r_image.shape}, {disp.shape}, [{flatten_key_to_fetch} : {pattern_to_fetch}]"
        sample = {
            'left': l_image * 255.,  # back to (0,255) for transform to do data augmentation.
            'right': r_image * 255,
            'disp': disp
        }
        sample = self.transform(sample)
        sample['index'] = index
        sample['name'] = flatten_key_to_fetch

        return sample


    def convert_imgs_to_gray(self, data):
        for k in data.keys():
            if 'Image' in k:
                img = data[k]
                if img.ndim == 3 and img.shape[2] == 3:
                    gray = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
                    data[k] = np.repeat(gray.astype(img.dtype)[..., None], 3, axis=-1)  # back to 3 channels but with gray values, for easier transform.
                elif img.ndim == 3 and img.shape[2] == 1:
                    data[k] = np.repeat(img, 3, axis=-1)  # back to 3 channels, for easier transform.
                elif img.ndim == 2:
                    data[k] = np.repeat(img[..., None], 3, axis=-1)  # back to 3 channels, for easier transform.
                else:
                    raise ValueError(f"Unsupported image shape: {img.shape}")
        return data