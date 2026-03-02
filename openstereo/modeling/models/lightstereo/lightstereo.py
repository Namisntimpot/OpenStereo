import torch.nn as nn
import torch.nn.functional as F
from openstereo.modeling.common.basic_block_2d import BasicConv2d, BasicDeconv2d
from openstereo.modeling.cost_volume.cost_volume import correlation_volume
from openstereo.modeling.disp_pred.disp_regression import disparity_regression
from openstereo.modeling.disp_refinement.disp_refinement import context_upsample

from .backbone import Backbone, FPNLayer
from .aggregation import Aggregation


class LightStereo(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.max_disp = cfgs.MAX_DISP
        self.left_att = cfgs.LEFT_ATT

        # backbobe
        self.backbone = Backbone(cfgs.get('BACKCONE', 'MobileNetv2'), pretrained=cfgs.get('PRETRAINED', True), pretrained_path=cfgs.get('PRETRAINED_PATH', None))

        # aggregation
        self.cost_agg = Aggregation(in_channels=48,
                                    left_att=self.left_att,
                                    blocks=cfgs.AGGREGATION_BLOCKS,
                                    expanse_ratio=cfgs.EXPANSE_RATIO,
                                    backbone_channels=self.backbone.output_channels)

        # disp refine
        self.refine_1 = nn.Sequential(
            BasicConv2d(self.backbone.output_channels[0], 24, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.InstanceNorm2d, act_layer=nn.LeakyReLU),
            BasicConv2d(24, 24, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.InstanceNorm2d, act_layer=nn.ReLU))

        self.stem_2 = nn.Sequential(
            BasicConv2d(3, 16, kernel_size=3, stride=2, padding=1,
                        norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU),
            BasicConv2d(16, 16, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU))
        self.refine_2 = FPNLayer(24, 16)

        self.refine_3 = BasicDeconv2d(16, 9, kernel_size=4, stride=2, padding=1)

        if cfgs.get("CUSTOM_INIT", False):
            for m in self.backbone.modules():
                m._is_backbone = True  # 标记backbone模块
            if not cfgs.get('PRETRAINED', True):
                self._custom_init_backbone()
            self._init_non_backbone_modules()
            print("[INIT] Initialized with MobileNetV2-aware strategy")

    def forward(self, data):
        image1 = data['left']
        image2 = data['right']

        features_left = self.backbone(image1)
        features_right = self.backbone(image2)

        gwc_volume = correlation_volume(features_left[0], features_right[0], self.max_disp // 4)
        encoding_volume = self.cost_agg(gwc_volume, features_left)  # [bz, 1, max_disp/4, H/4, W/4]
        squeezed_encoding = encoding_volume[0].reshape(encoding_volume[0].size(0), -1, encoding_volume[0].size(2), encoding_volume[0].size(3))  # [bz, max_disp/4, H/4, W/4]

        prob = F.softmax(squeezed_encoding, dim=1)
        init_disp = disparity_regression(prob, self.max_disp // 4)  # [bz, 1, H/4, W/4]

        xspx = self.refine_1(features_left[0])
        xspx = self.refine_2(xspx, self.stem_2(image1))
        xspx = self.refine_3(xspx)
        spx_pred = F.softmax(xspx, 1)  # [bz, 9, H, W]
        disp_pred = context_upsample(init_disp * 4., spx_pred.float()).unsqueeze(1)  # # [bz, 1, H, W]

        result = {'disp_pred': disp_pred}

        if self.training:
            disp_4 = F.interpolate(init_disp, image1.shape[2:], mode='bilinear', align_corners=False)
            disp_4 *= 4
            result['disp_4'] = disp_4

        return result

    def get_loss(self, model_pred, input_data):
        disp_gt = input_data["disp"]  # [bz, h, w]
        disp_gt = disp_gt.unsqueeze(1)  # [bz, 1, h, w]
        mask = (disp_gt < self.max_disp) & (disp_gt > 0)  # [bz, 1, h, w]

        disp_pred = model_pred['disp_pred']
        loss = 1.0 * F.smooth_l1_loss(disp_pred[mask], disp_gt[mask], reduction='mean')

        disp_4 = model_pred['disp_4']
        loss += 0.3 * F.smooth_l1_loss(disp_4[mask], disp_gt[mask], reduction='mean')

        loss_info = {'scalar/train/loss_disp': loss.item()}

        return loss, loss_info


    def _init_non_backbone_modules(self):
        """初始化除backbone外的所有模块"""
        def init_module(m):
            # 跳过backbone和已初始化模块
            if getattr(m, '_is_backbone', False) or hasattr(m, '_custom_initialized'):
                return
            
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                # Depthwise卷积特殊处理
                if m.groups == m.in_channels and m.in_channels > 1:
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                else:  # 标准/pointwise卷积
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                m._custom_initialized = True
                
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                m._custom_initialized = True
                
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.InstanceNorm2d, nn.InstanceNorm1d)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
                m._custom_initialized = True
                
            elif isinstance(m, nn.GroupNorm):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None: 
                    nn.init.zeros_(m.bias)
                m._custom_initialized = True
        
        # 递归应用初始化
        self.apply(init_module)
        
        # 标记顶层模块已初始化
        self._non_backbone_initialized = True

    def _custom_init_backbone(self):
        """MobileNetV2定制化初始化"""
        def init_mbnet_module(m):
            if isinstance(m, nn.Conv2d):
                # Depthwise卷积 (groups == in_channels 且 in_channels > 1)
                if m.groups == m.in_channels and m.in_channels > 1:
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                else:  # Pointwise (1x1) 或 标准卷积
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            
            # SE模块特殊处理（若存在）
            elif hasattr(m, 'fc1') and hasattr(m, 'fc2'):  # 简易检测SE模块
                if hasattr(m.fc2, 'bias'):
                    nn.init.constant_(m.fc2.bias, -3.0)  # 使Sigmoid初始输出≈0.05
        
        self.backbone.apply(init_mbnet_module)