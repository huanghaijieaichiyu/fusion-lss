import torch
from torch import nn
from torchvision.models.resnet import resnet18
import timm
import torch.nn.functional as F
import os

from .tools import gen_dx_bx, cumsum_trick, QuickCumsum

# 添加EfficientNet的条件导入
try:
    from efficientnet_pytorch import EfficientNet
except ImportError:
    print("警告: efficientnet_pytorch 未安装。如需使用 CamEncode 类，请先安装: pip install efficientnet-pytorch")
    # 创建一个空的EfficientNet类，以避免导入错误

    class EfficientNet:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise ImportError(
                "未安装 efficientnet_pytorch 模块。请使用 pip install efficientnet-pytorch 安装。")


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class CamEncode(nn.Module):
    def __init__(self, D, C, downsample):
        super(CamEncode, self).__init__()
        self.D = D
        self.C = C

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")

        self.up1 = Up(320+112, 512)
        self.depthnet = nn.Conv2d(
            512, self.D + self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, x):
        x = self.get_eff_depth(x)
        # Depth
        x = self.depthnet(x)

        depth = self.get_depth_dist(x[:, :self.D])
        new_x = depth.unsqueeze(
            1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)

        return depth, new_x

    def get_eff_depth(self, x):
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                # scale drop connect_rate
                drop_connect_rate *= float(idx) / len(self.trunk._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            prev_x = x

        # Head
        endpoints['reduction_{}'.format(len(endpoints)+1)] = x
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])
        return x

    def forward(self, x):
        depth, x = self.get_depth_feat(x)

        return x


class BevEncode(nn.Module):
    def __init__(self, inC, outC):
        super(BevEncode, self).__init__()

        trunk = resnet18(weights=None, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64+256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)

        x = self.up1(x, x1)
        x = self.up2(x)

        return x


class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio=0.0):  # Default se_ratio to 0
        super().__init__()
        self.stride = stride
        # Ensure hidden_dim is an integer
        hidden_dim = int(in_channels * expand_ratio)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        # Expansion phase
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim,
                          kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))

        # Depthwise convolution
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride,
                                padding=(kernel_size - 1) // 2, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU6(inplace=True))

        # Squeeze-and-Excite (Placeholder for future, currently se_ratio=0 means disabled)
        # if se_ratio > 0:
        #     num_squeezed_channels = max(1, int(in_channels * se_ratio)) # Squeeze from in_channels
        #     # SELayer would need to be defined and added here, operating on hidden_dim
        #     pass

        # Projection phase (linear)
        layers.append(nn.Conv2d(hidden_dim, out_channels,
                      kernel_size=1, bias=False))
        # No activation after projection if it's the end of the block before residual
        layers.append(nn.BatchNorm2d(out_channels))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class BevEncodeV2(nn.Module):
    def __init__(self, inC, outC):
        super(BevEncodeV2, self).__init__()

        # Configuration for MBConv stages
        # (input_channels, output_channels, num_blocks, stride_first_block, expand_ratio, kernel_size)
        # Channels based on a light configuration, e.g., ~MobileNetV2-like
        c_stem = 32
        stage_configs = [
            (c_stem, 48, 2, 2, 6, 3),    # stage1: H/2 -> H/4
            (48, 64, 2, 2, 6, 3),     # stage2: H/4 -> H/8
            (64, 96, 2, 1, 6, 3),     # stage3: H/8 -> H/8 (more features)
        ]

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(inC, c_stem, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c_stem),
            nn.ReLU6(inplace=True)
        )
        self.s1_feats_ch = c_stem  # Channels after stem, input to stage1

        # Downsampling Stages (Encoder)
        self.stages = nn.ModuleList()
        current_channels = c_stem
        # To store channels of skip connections
        self.skip_channels = [current_channels]

        for C_in_stage_def_unused, C_out_stage, num_blocks, stride_first, exp_r, ks in stage_configs:
            stage_blocks = []
            for i in range(num_blocks):
                stride = stride_first if i == 0 else 1
                stage_blocks.append(
                    MBConvBlock(current_channels, C_out_stage,
                                ks, stride, exp_r)
                )
                current_channels = C_out_stage
            self.stages.append(nn.Sequential(*stage_blocks))
            # Store output channels of this stage
            self.skip_channels.append(current_channels)

        # self.skip_channels will be [c_stem, stage1_out_ch, stage2_out_ch, stage3_out_ch]
        # Example: [32, 48, 64, 96]

        # Upsampling Path (Decoder) - correcting skip connections
        up_c1 = 64
        up_c2 = 48

        # up1: Upsamples stage3_out (x1, H/8) to H/4, concatenates with stage1_out (x2, H/4)
        # Input channels to Up.conv = ch(stage3) + ch(stage1)
        self.up1 = Up(
            self.skip_channels[3] + self.skip_channels[1], up_c1, scale_factor=2)

        # up2: Upsamples up1_out (x1, H/4) to H/2, concatenates with stem_out (x2, H/2)
        # Input channels to Up.conv = ch(up1_out) + ch(stem)
        self.up2 = Up(up_c1 + self.skip_channels[0], up_c2, scale_factor=2)

        # Final upsampling and convolution, similar to original BevEncode.up2
        # Input is up2_out (d2), spatial H/2, channels up_c2
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),  # H/2 -> H
            nn.Conv2d(up_c2, up_c2, kernel_size=3, padding=1,
                      bias=False),  # Input from up2
            nn.BatchNorm2d(up_c2),
            nn.ReLU6(inplace=True),
            nn.Conv2d(up_c2, outC, kernel_size=1, padding=0)
        )

    def forward(self, x):
        # Stem
        s1_f = self.stem(x)  # H/2, W/2 ; Channels: self.skip_channels[0]

        # Encoder Path
        skip_connections = [s1_f]
        f_current = s1_f
        for stage_module in self.stages:
            f_current = stage_module(f_current)
            skip_connections.append(f_current)

        # skip_connections indices: [0: stem_out(H/2), 1: stage1_out(H/4), 2: stage2_out(H/8), 3: stage3_out(H/8)]

        # Decoder Path
        # d1 = up1(stage3_out, stage1_out)
        # Output: up_c1 channels, H/4
        d1 = self.up1(skip_connections[3], skip_connections[1])

        # d2 = up2(d1, stem_out)
        d2 = self.up2(d1, skip_connections[0])  # Output: up_c2 channels, H/2

        # Final upsampling and output conv
        out = self.final_up(d2)  # Output: outC channels, H, W

        return out


class LiftSplatShoot(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, outC):
        super(LiftSplatShoot, self).__init__()
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf

        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                               self.grid_conf['ybound'],
                               self.grid_conf['zbound'],
                               )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.downsample = 16
        self.camC = 64
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
        self.camencode = CamEncode(self.D, self.camC, self.downsample)
        self.bevencode = BevEncode(inC=self.camC, outC=outC)

        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True

    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf['final_dim']
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(
            *self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(
            0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(
            0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape

        # 初始点投影
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)  # Step 1

        # 计算逆矩阵并进行投影
        inverse_epsilon = 1e-6  # Small value for stability
        # Add epsilon * I to post_rots before inversion
        stable_post_rots = post_rots + \
            torch.eye(3, device=post_rots.device, dtype=post_rots.dtype).unsqueeze(
                0).unsqueeze(0) * inverse_epsilon
        inv_post_rots = torch.inverse(stable_post_rots)  # Step 2a

        points = inv_post_rots.view(
            B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))  # Step 2b

        # 关键修改: 添加数值稳定性
        xy_coords = points[:, :, :, :, :, :2]
        depth_vals = points[:, :, :, :, :, 2:3]
        epsilon = 1e-6  # This epsilon is for safe_depth, not the inverse_epsilon
        safe_depth = depth_vals.clone()
        safe_depth = torch.where(torch.abs(safe_depth) < epsilon,
                                 torch.full_like(
                                     safe_depth, epsilon) * torch.sign(safe_depth + epsilon),
                                 safe_depth)

        # --- Perform critical multiplication in float32 to prevent float16 overflow ---
        product_f32 = xy_coords.float() * safe_depth.float()

        # Ensure the other part for concatenation is also float32
        safe_depth_component_f32 = safe_depth.float()
        # ---------------------------------------------------------------------------

        # Step 3 (Scaling)
        # Resulting 'points' will be float32 due to product_f32 and safe_depth_component_f32
        points = torch.cat(
            (product_f32, safe_depth_component_f32), 5)

        # 应用相机到自车坐标的变换
        # Add epsilon * I to intrins before inversion
        stable_intrins = intrins + \
            torch.eye(3, device=intrins.device, dtype=intrins.dtype).unsqueeze(
                0).unsqueeze(0) * inverse_epsilon
        inv_intrins = torch.inverse(stable_intrins)  # Step 4a

        combine = rots.matmul(inv_intrins)  # Step 4b

        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(
            points.squeeze(-1).unsqueeze(-1)).squeeze(-1)  # Step 4c (Corrected matmul)

        # 添加平移
        points += trans.view(B, N, 1, 1, 1, 3)  # Step 5

        return points  # Final Output

    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape

        x = x.view(B*N, C, imH, imW)
        x = self.camencode(x)
        x = x.view(B, N, self.camC, self.D, imH //
                   self.downsample, imW//self.downsample)
        x = x.permute(0, 1, 3, 4, 5, 2)

        return x

    def voxel_pooling(self, geom_feats, x):
        """Pools camera features into a BEV grid.
        Args:
            geom_feats: Geometry tensor (B x N x D x H x W x 3)
            x: Camera features (B x N x D x H x W x C)
        Returns:
            BEV features (B x (C * Z) x X x Y)
        """
        B, N, D, H, W, C = x.shape
        Nprime = B*N*D*H*W

        # 展平特征
        x = x.reshape(Nprime, C)

        # 计算栅格化索引
        geom_feats_normalized = (geom_feats - (self.bx - self.dx/2.)) / self.dx
        geom_feats_long = geom_feats_normalized.long()
        geom_feats = geom_feats_long.view(Nprime, 3)

        # 添加批次索引
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # 过滤掉边界外的点
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])

        # 改进的处理方式：处理没有有效点的情况
        if not torch.any(kept):
            # 使用中心区域的一些点，即使它们可能在边界外
            # 我们只选择一个小的中心区域（不超过10%的原始点）
            center_region = (
                torch.abs(geom_feats_normalized[:, 0] - self.nx[0]/2) < self.nx[0]*0.2) & \
                (torch.abs(geom_feats_normalized[:, 1] - self.nx[1]/2) < self.nx[1]*0.2) & \
                (torch.abs(
                    geom_feats_normalized[:, 2] - self.nx[2]/2) < self.nx[2]*0.2)

            # 若中心区域也没有点，则选择最接近中心的几个点
            if not torch.any(center_region):
                # 计算到中心的距离
                center_distances = (geom_feats_normalized[:, 0] - self.nx[0]/2)**2 + \
                    (geom_feats_normalized[:, 1] - self.nx[1]/2)**2 + \
                    (geom_feats_normalized[:,
                                           2] - self.nx[2]/2)**2
                # 选择最近的10个点
                _, closest_indices = torch.topk(center_distances, k=min(
                    10, center_distances.size(0)), largest=False)
                new_kept = torch.zeros_like(kept)
                new_kept[closest_indices] = True
                kept = new_kept
            else:
                # 使用中心区域点
                kept = center_region

            # 将这些点映射到有效范围内
            geom_feats[:, 0] = torch.clamp(
                geom_feats[:, 0], 0, int(self.nx[0].item()-1))
            geom_feats[:, 1] = torch.clamp(
                geom_feats[:, 1], 0, int(self.nx[1].item()-1))
            geom_feats[:, 2] = torch.clamp(
                geom_feats[:, 2], 0, int(self.nx[2].item()-1))

        x = x[kept]
        geom_feats = geom_feats[kept]

        # 计算体素排序索引
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]

        # 对点进行排序以使相同体素的点相邻
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # 执行累加技巧 - 始终使用cumsum_trick（FusionNet中self.use_quickcumsum = False）
        x, geom_feats = cumsum_trick(x, geom_feats, ranks)

        # 创建最终的BEV网格
        final = torch.zeros(
            (B, C, int(self.nx[2].item()), int(self.nx[0].item()), int(self.nx[1].item())), device=x.device)

        # 将处理后的特征放入网格中
        final[geom_feats[:, 3], :, geom_feats[:, 2],
              geom_feats[:, 0], geom_feats[:, 1]] = x

        return final

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        x = self.get_cam_feats(x)

        x = self.voxel_pooling(geom, x)

        return x

    def forward(self, x, rots, trans, intrins, post_rots, post_trans):
        x = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)
        x = self.bevencode(x)
        return x


class CamEncodeFPN(nn.Module):
    def __init__(self, D, C, fpn_out_channels=256):
        super(CamEncodeFPN, self).__init__()
        self.D = D
        self.C = C
        self.fpn_out_channels = fpn_out_channels

        # --- 使用 RegNetY-400MF 作为 Backbone ---
        self.trunk = timm.create_model(
            'regnety_004',
            pretrained=False,  # 确保不加载任何预训练权重
            features_only=True,
            out_indices=[1, 2, 3, 4]
        )

        # 在模型创建后手动加载并处理权重
        checkpoint_path = 'weights/pytorch_model.bin'
        if os.path.exists(checkpoint_path):
            try:
                state_dict = torch.load(checkpoint_path, map_location='cpu')

                # 处理不同格式的检查点文件
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                elif 'model' in state_dict:
                    state_dict = state_dict['model']

                # 获取 trunk 模型期望的 state_dict 键
                trunk_keys = self.trunk.state_dict().keys()

                # 过滤掉不属于 trunk 的键，并处理 'module.' 前缀
                filtered_state_dict = {}
                ignored_keys_count = 0
                loaded_keys_count = 0
                for k, v in state_dict.items():
                    # 检查原始键或移除 'module.' 前缀后的键是否存在于 trunk 中
                    plain_k = k.replace('module.', '', 1)
                    if plain_k in trunk_keys:
                        filtered_state_dict[plain_k] = v
                        loaded_keys_count += 1
                    else:
                        ignored_keys_count += 1

                # 使用strict=True加载过滤后的权重，因为我们只保留了期望的键
                missing_keys, unexpected_keys = self.trunk.load_state_dict(
                    filtered_state_dict, strict=True)

                print(
                    f"权重加载: 加载 {loaded_keys_count} 个键, 忽略 {ignored_keys_count} 个键.")
                if missing_keys or unexpected_keys:
                    print(
                        f"  警告: 加载权重后发现缺失或意外的键。缺失: {missing_keys}, 意外: {unexpected_keys}")

            except Exception as e:
                print(f"加载权重时出错: {e}")
                print(f"将使用随机初始化的权重继续")
        else:
            print(f"警告: 在{checkpoint_path}未找到检查点文件。将使用随机初始化的权重。")

        # --- 获取特征通道信息 ---
        feature_info = self.trunk.feature_info.get_dicts(
            keys=['num_chs', 'reduction'])
        if len(feature_info) < 4:
            raise ValueError(
                f"timm模型'regnety_004'未返回预期数量的特征阶段 (预期4个, 得到{len(feature_info)}个)")

        fpn_channels = [info['num_chs'] for info in feature_info]

        # --- FPN侧向连接 ---
        self.lat_c2 = nn.Conv2d(
            fpn_channels[0], self.fpn_out_channels, kernel_size=1)
        self.lat_c3 = nn.Conv2d(
            fpn_channels[1], self.fpn_out_channels, kernel_size=1)
        self.lat_c4 = nn.Conv2d(
            fpn_channels[2], self.fpn_out_channels, kernel_size=1)
        self.lat_c5 = nn.Conv2d(
            fpn_channels[3], self.fpn_out_channels, kernel_size=1)

        # --- FPN自顶向下路径 ---
        self.smooth_p4 = nn.Sequential(
            nn.Conv2d(self.fpn_out_channels, self.fpn_out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.fpn_out_channels),
            nn.ReLU(inplace=True)
        )
        self.smooth_p3 = nn.Sequential(
            nn.Conv2d(self.fpn_out_channels, self.fpn_out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.fpn_out_channels),
            nn.ReLU(inplace=True)
        )
        self.smooth_p2 = nn.Sequential(
            nn.Conv2d(self.fpn_out_channels, self.fpn_out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.fpn_out_channels),
            nn.ReLU(inplace=True)
        )

        # --- 改进的深度头 (输出不确定性, 从P2输入) ---
        depth_intermediate_channels = self.fpn_out_channels // 2
        self.depthnet = nn.Sequential(
            nn.Conv2d(self.fpn_out_channels, depth_intermediate_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(depth_intermediate_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(depth_intermediate_channels, depth_intermediate_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(depth_intermediate_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(depth_intermediate_channels, 2 *
                      self.D + self.C, kernel_size=1, padding=0)
        )

        # --- 添加深度细化模块 (Input from P2) ---
        self.refinement_net = DepthRefinementNetV2(
            D, self.fpn_out_channels, intermediate_channels=fpn_out_channels // 2)

    def get_depth_dist(self, depth_logits, eps=1e-20):
        return depth_logits.softmax(dim=1)

    def forward(self, x):
        # 检查 CamEncodeFPN 的输入
        # assert torch.all(torch.isfinite(
        #     x)), "NaN/Inf detected in input x to CamEncodeFPN forward"

        # --- Backbone Feature Extraction ---
        features = self.trunk(x)
        if len(features) != 4:
            raise ValueError(
                f"Expected 4 feature maps from backbone, got {len(features)}")
        c2, c3, c4, c5 = features
        # 检查骨干网输出的关键部分
        # assert torch.all(torch.isfinite(
        #     c2)), "NaN/Inf detected in c2 from backbone"
        # assert torch.all(torch.isfinite(
        #     c5)), "NaN/Inf detected in c5 from backbone"

        # --- FPN Lateral Connections and Top-Down Pathway ---
        # 逐层检查 FPN 输出
        p5 = self.lat_c5(c5)
        # assert torch.all(torch.isfinite(p5)), "NaN/Inf detected in p5"
        p4_in = self.lat_c4(
            c4) + F.interpolate(p5, size=c4.shape[2:], mode='bilinear', align_corners=True)
        # assert torch.all(torch.isfinite(p4_in)), "NaN/Inf detected in p4_in"
        p4 = self.smooth_p4(p4_in)
        # assert torch.all(torch.isfinite(p4)), "NaN/Inf detected in p4"

        p3_in = self.lat_c3(
            c3) + F.interpolate(p4, size=c3.shape[2:], mode='bilinear', align_corners=True)
        # assert torch.all(torch.isfinite(p3_in)), "NaN/Inf detected in p3_in"
        p3 = self.smooth_p3(p3_in)
        # assert torch.all(torch.isfinite(p3)), "NaN/Inf detected in p3"

        p2_in = self.lat_c2(
            c2) + F.interpolate(p3, size=c2.shape[2:], mode='bilinear', align_corners=True)
        # assert torch.all(torch.isfinite(p2_in)), "NaN/Inf detected in p2_in"
        p2 = self.smooth_p2(p2_in)
        # assert torch.all(torch.isfinite(p2)), "NaN/Inf detected in p2"

        # --- 设定 log_variance 的安全范围 ---
        log_var_min = -10.0
        log_var_max = 10.0

        # --- Initial Depth, Uncertainty, and Feature Calculation from P2 ---
        output = self.depthnet(p2)
        # --- 检查 depthnet 的整体输出 ---
        # assert torch.all(torch.isfinite(
        #     output)), "!!! NaN/Inf detected immediately after self.depthnet(p2) call !!!"
        # ---------------------------------
        initial_depth_logits = output[:, :self.D]
        initial_depth_log_variance_raw = output[:, self.D:2*self.D]
        context_features = output[:, 2*self.D:]
        # --- 检查 depthnet 输出的分割部分 ---
        # assert torch.all(torch.isfinite(initial_depth_logits)
        #                  ), "NaN/Inf detected in initial_depth_logits split from depthnet output"
        # assert torch.all(torch.isfinite(initial_depth_log_variance_raw)
        #                  ), "NaN/Inf detected in initial_depth_log_variance_raw split from depthnet output"
        # assert torch.all(torch.isfinite(
        #     context_features)), "NaN/Inf detected in context_features split from depthnet output"
        # ----------------------------------

        # --- 稳定化: Clamp 初始 log_variance ---
        initial_depth_log_variance = torch.clamp(
            initial_depth_log_variance_raw, min=log_var_min, max=log_var_max
        )
        # assert torch.all(torch.isfinite(initial_depth_log_variance)
        #                  ), "NaN/Inf detected after clamping initial_depth_log_variance"

        # --- Depth Refinement using P2 features ---
        # 检查 refinement_net 的输入
        # assert torch.all(torch.isfinite(initial_depth_logits.detach(
        # ))), "NaN/Inf in input initial_logits.detach() to refinement_net"
        # assert torch.all(torch.isfinite(initial_depth_log_variance.detach(
        # ))), "NaN/Inf in input initial_depth_log_variance.detach() to refinement_net"
        # assert torch.all(torch.isfinite(
        #     p2)), "NaN/Inf in input p2 to refinement_net"

        refined_depth_logits_raw, refined_depth_log_variance_raw = self.refinement_net(
            initial_depth_logits.detach(),
            initial_depth_log_variance.detach(),  # 使用 clamp 后的值 detach
            p2
        )
        # --- 检查 refinement_net 的输出 ---
        # assert torch.all(torch.isfinite(refined_depth_logits_raw)
        #                  ), "!!! NaN/Inf detected in refined_depth_logits_raw output from refinement_net !!!"
        # assert torch.all(torch.isfinite(refined_depth_log_variance_raw)
        #                  ), "!!! NaN/Inf detected in refined_depth_log_variance_raw output from refinement_net !!!"
        # ---------------------------------

        # --- 稳定化: Clamp 细化后的 log_variance ---
        refined_depth_logits = refined_depth_logits_raw  # 使用 refinement_net 的原始 logits 输出
        refined_depth_log_variance = torch.clamp(
            refined_depth_log_variance_raw, min=log_var_min, max=log_var_max
        )
        # --- 检查 clamp 后的值 ---
        # assert torch.all(torch.isfinite(refined_depth_logits)
        #                  ), "NaN/Inf detected in refined_depth_logits after assignment (before softmax)"
        # assert torch.all(torch.isfinite(refined_depth_log_variance)
        #                  ), "NaN/Inf detected after clamping refined_depth_log_variance"
        # ---------------------------

        # --- Final Feature Combination using Refined Depth ---
        refined_depth_prob = self.get_depth_dist(refined_depth_logits)
        # --- 检查 softmax 输出 ---
        # assert torch.all(torch.isfinite(refined_depth_prob)
        #                  ), "!!! NaN/Inf detected in refined_depth_prob after softmax !!!"
        # --------------------------

        refined_confidence = torch.exp(-refined_depth_log_variance)
        # --- 检查 exp 输出 ---
        # assert torch.all(torch.isfinite(refined_confidence)
        #                  ), "!!! NaN/Inf detected in refined_confidence after exp(-log_var) !!!"
        # --------------------

        # 检查最终乘法的输入
        # assert torch.all(torch.isfinite(refined_depth_prob.unsqueeze(
        #     1))), "NaN/Inf in input 1 (prob) to final multiplication"
        # assert torch.all(torch.isfinite(refined_confidence.unsqueeze(
        #     1))), "NaN/Inf in input 2 (conf) to final multiplication"
        # assert torch.all(torch.isfinite(context_features.unsqueeze(
        #     2))), "NaN/Inf in input 3 (feat) to final multiplication"

        epsilon = 1e-6
        new_x = refined_depth_prob.unsqueeze(
            1) * refined_confidence.unsqueeze(1) * context_features.unsqueeze(2) + epsilon
        # 检查最终输出 new_x
        # assert torch.all(torch.isfinite(
        #     new_x)), "!!! NaN/Inf detected in the final new_x output of CamEncodeFPN !!!"

        return new_x, refined_depth_prob


class LidarEncode(nn.Module):
    def __init__(self, inC_lidar, outC_lidar):
        super().__init__()
        # Example: A simple ConvNet encoder for LiDAR BEV features
        self.encoder = nn.Sequential(
            nn.Conv2d(inC_lidar, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, outC_lidar, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(outC_lidar),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.encoder(x)


class FusionNet(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, outC, lidar_inC=1, lidar_enc_out_channels=128, fused_bev_channels=256):
        super(FusionNet, self).__init__()
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf

        # --- Grid and Frustum Setup (Moved from LiftSplatShoot) ---
        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                               self.grid_conf['ybound'],
                               self.grid_conf['zbound'],
                               )
        # Ensure nx are integers for grid creation
        nx = nx.long()
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        # Determine frustum based on FPN output stride
        # Use stride 4 (from p2) to match get_cam_feats output
        self.feature_map_stride = 4
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
        # --- Grid and Frustum End ---

        # --- Camera Branch Components ---
        fpn_feature_channels = 256  # Corresponds to FPN output channels
        context_channels_per_depth_bin = 64  # Desired C per depth bin for voxel pooling
        self.camC = context_channels_per_depth_bin  # Channels per depth bin feature
        self.camencode = CamEncodeFPN(
            self.D, self.camC, fpn_out_channels=fpn_feature_channels)
        # --- Camera Branch End ---

        # --- LiDAR Branch Components ---
        self.lidar_enc_out_channels = lidar_enc_out_channels
        self.lidarencode = LidarEncode(lidar_inC, self.lidar_enc_out_channels)
        # --- LiDAR Branch End ---

        # --- Fusion and BEV Encoding ---
        num_z_bins = self.nx[2].item()  # Get integer Z dimension
        # Input channels for camera BEV processor after Z-collapse (concatenation)
        cam_bev_interim_channels = self.camC * num_z_bins

        # Camera BEV Processor (using BevEncodeV2 architecture)
        self.cam_bev_processor = BevEncodeV2(
            inC=cam_bev_interim_channels, outC=fused_bev_channels)

        # Fusion layer: combines processed camera BEV and processed lidar BEV
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fused_bev_channels + self.lidar_enc_out_channels,
                      fused_bev_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fused_bev_channels),
            nn.ReLU(inplace=True),
            # Final output layer
            nn.Conv2d(fused_bev_channels, outC, kernel_size=1, padding=0)
        )
        # --- Fusion and BEV End ---

        # 设置使用cumsum_trick，不使用QuickCumsum（保持一致性并确保安全）
        self.use_quickcumsum = False

    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf['final_dim']
        # Use self.feature_map_stride determined in __init__
        fH, fW = ogfH // self.feature_map_stride, ogfW // self.feature_map_stride
        ds = torch.arange(
            *self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(
            0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(
            0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape

        # 初始点投影
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)  # Step 1

        # 计算逆矩阵并进行投影
        inverse_epsilon = 1e-6  # Small value for stability
        # Add epsilon * I to post_rots before inversion
        stable_post_rots = post_rots + \
            torch.eye(3, device=post_rots.device, dtype=post_rots.dtype).unsqueeze(
                0).unsqueeze(0) * inverse_epsilon
        inv_post_rots = torch.inverse(stable_post_rots)  # Step 2a

        points = inv_post_rots.view(
            B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))  # Step 2b

        # 关键修改: 添加数值稳定性
        xy_coords = points[:, :, :, :, :, :2]
        depth_vals = points[:, :, :, :, :, 2:3]
        epsilon = 1e-6  # This epsilon is for safe_depth, not the inverse_epsilon
        safe_depth = depth_vals.clone()
        safe_depth = torch.where(torch.abs(safe_depth) < epsilon,
                                 torch.full_like(
                                     safe_depth, epsilon) * torch.sign(safe_depth + epsilon),
                                 safe_depth)

        # --- Perform critical multiplication in float32 to prevent float16 overflow ---
        product_f32 = xy_coords.float() * safe_depth.float()

        # Ensure the other part for concatenation is also float32
        safe_depth_component_f32 = safe_depth.float()
        # ---------------------------------------------------------------------------

        # Step 3 (Scaling)
        # Resulting 'points' will be float32 due to product_f32 and safe_depth_component_f32
        points = torch.cat(
            (product_f32, safe_depth_component_f32), 5)

        # 应用相机到自车坐标的变换
        # Add epsilon * I to intrins before inversion
        stable_intrins = intrins + \
            torch.eye(3, device=intrins.device, dtype=intrins.dtype).unsqueeze(
                0).unsqueeze(0) * inverse_epsilon
        inv_intrins = torch.inverse(stable_intrins)  # Step 4a

        combine = rots.matmul(inv_intrins)  # Step 4b

        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(
            points.squeeze(-1).unsqueeze(-1)).squeeze(-1)  # Step 4c (Corrected matmul)

        # 添加平移
        points += trans.view(B, N, 1, 1, 1, 3)  # Step 5

        return points  # Final Output

    def get_cam_feats(self, x):
        """Return features: B x N x D x H_feat x W_feat x C_feat (camC)
                 depth_prob: B x N x D x H_feat x W_feat
        """
        B, N, C_img, imH, imW = x.shape
        x_batch_n = x.view(B*N, C_img, imH, imW)

        # CamEncodeFPN returns (features, depth_prob)
        # Features shape: (B*N, C_feat, D, H_feat, W_feat) where C_feat = self.camC
        # Depth prob shape: (B*N, D_feat, H_feat, W_feat)
        features_bn, depth_prob_bn = self.camencode(x_batch_n)

        # Reshape and permute features for voxel_pooling
        _, C_feat, D_feat, H_feat, W_feat = features_bn.shape
        assert D_feat == self.D, f"Depth dim mismatch: CamEncode output {D_feat} vs Frustum {self.D}"
        assert C_feat == self.camC, f"Channel dim mismatch: CamEncode output {C_feat} vs self.camC {self.camC}"

        features_bn = features_bn.view(B, N, C_feat, D_feat, H_feat, W_feat)
        # Permute features to B x N x D x H x W x C for voxel_pooling
        features_permuted = features_bn.permute(0, 1, 3, 4, 5, 2)

        # Reshape depth probability
        depth_prob = depth_prob_bn.view(B, N, D_feat, H_feat, W_feat)

        return features_permuted, depth_prob  # 返回特征和深度概率

    def voxel_pooling(self, geom_feats, x):
        """Pools camera features into a BEV grid.
        Args:
            geom_feats: Geometry tensor (B x N x D x H x W x 3)
            x: Camera features (B x N x D x H x W x C)
        Returns:
            BEV features (B x (C * Z) x X x Y)
        """
        B, N, D, H, W, C = x.shape
        Nprime = B*N*D*H*W

        # 展平特征
        x = x.reshape(Nprime, C)

        # 计算栅格化索引
        geom_feats_normalized = (geom_feats - (self.bx - self.dx/2.)) / self.dx
        geom_feats_long = geom_feats_normalized.long()
        geom_feats = geom_feats_long.view(Nprime, 3)

        # 添加批次索引
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # 过滤掉边界外的点
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])

        # 改进的处理方式：处理没有有效点的情况
        if not torch.any(kept):
            # 使用中心区域的一些点，即使它们可能在边界外
            # 我们只选择一个小的中心区域（不超过10%的原始点）
            center_region = (
                torch.abs(geom_feats_normalized[:, 0] - self.nx[0]/2) < self.nx[0]*0.2) & \
                (torch.abs(geom_feats_normalized[:, 1] - self.nx[1]/2) < self.nx[1]*0.2) & \
                (torch.abs(
                    geom_feats_normalized[:, 2] - self.nx[2]/2) < self.nx[2]*0.2)

            # 若中心区域也没有点，则选择最接近中心的几个点
            if not torch.any(center_region):
                # 计算到中心的距离
                center_distances = (geom_feats_normalized[:, 0] - self.nx[0]/2)**2 + \
                    (geom_feats_normalized[:, 1] - self.nx[1]/2)**2 + \
                    (geom_feats_normalized[:,
                                           2] - self.nx[2]/2)**2
                # 选择最近的10个点
                _, closest_indices = torch.topk(center_distances, k=min(
                    10, center_distances.size(0)), largest=False)
                new_kept = torch.zeros_like(kept)
                new_kept[closest_indices] = True
                kept = new_kept
            else:
                # 使用中心区域点
                kept = center_region

            # 将这些点映射到有效范围内
            geom_feats[:, 0] = torch.clamp(
                geom_feats[:, 0], 0, int(self.nx[0].item()-1))
            geom_feats[:, 1] = torch.clamp(
                geom_feats[:, 1], 0, int(self.nx[1].item()-1))
            geom_feats[:, 2] = torch.clamp(
                geom_feats[:, 2], 0, int(self.nx[2].item()-1))

        x = x[kept]
        geom_feats = geom_feats[kept]

        # 计算体素排序索引
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]

        # 对点进行排序以使相同体素的点相邻
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # 执行累加技巧 - 始终使用cumsum_trick（FusionNet中self.use_quickcumsum = False）
        x, geom_feats = cumsum_trick(x, geom_feats, ranks)

        # 创建最终的BEV网格
        final = torch.zeros(
            (B, C, int(self.nx[2].item()), int(self.nx[0].item()), int(self.nx[1].item())), device=x.device)

        # 将处理后的特征放入网格中
        final[geom_feats[:, 3], :, geom_feats[:, 2],
              geom_feats[:, 0], geom_feats[:, 1]] = x

        return final

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        """Gets camera features projected onto voxel grid and depth probabilities.
        Returns:
            cam_bev_interim: (B, C_interim, X, Y)
            depth_prob: (B, N, D, H_feat, W_feat)
        """
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        # Get features and depth probability
        cam_features_per_pixel, depth_prob = self.get_cam_feats(x)
        cam_bev_interim = self.voxel_pooling(geom, cam_features_per_pixel)
        return cam_bev_interim, depth_prob  # 返回 BEV 特征和深度概率

    def forward(self, x_cam, rots, trans, intrins, post_rots, post_trans, lidar_bev):
        """Forward pass for multi-modal fusion.
        Returns:
            output: Final BEV prediction (B, outC, H_bev, W_bev)
            depth_prob: Camera depth probability (B, N, D, H_feat, W_feat)
        """
        # 1. Get Camera BEV features and depth probability
        cam_bev_interim, depth_prob = self.get_voxels(  # 接收 depth_prob
            x_cam, rots, trans, intrins, post_rots, post_trans)
        # assert torch.all(torch.isfinite(cam_bev_interim)
        #                  ), "NaN/Inf detected in cam_bev_interim from get_voxels"
        # assert torch.all(torch.isfinite(depth_prob)
        #                  ), "NaN/Inf detected in depth_prob from get_voxels"

        # 2. Process Camera BEV features
        # --- Reshape cam_bev_interim before passing to processor ---
        B_cam, C_cam, Z_cam, X_cam, Y_cam = cam_bev_interim.shape
        cam_bev_interim_reshaped = cam_bev_interim.view(
            B_cam, C_cam * Z_cam, X_cam, Y_cam)
        # -----------------------------------------------------------
        cam_bev_processed = self.cam_bev_processor(
            cam_bev_interim_reshaped)  # 使用 reshape 后的张量
        # assert torch.all(torch.isfinite(cam_bev_processed)
        #                  ), "NaN/Inf detected after cam_bev_processor"

        # 3. Process LiDAR BEV features
        # assert torch.all(torch.isfinite(lidar_bev)
        #                  ), "NaN/Inf detected in lidar_bev input"
        lidar_bev_processed = self.lidarencode(lidar_bev)
        # assert torch.all(torch.isfinite(lidar_bev_processed)
        #                  ), "NaN/Inf detected after lidarencode"

        # 4. Fuse Features
        if cam_bev_processed.shape[2:] != lidar_bev_processed.shape[2:]:
            lidar_bev_processed_resized = F.interpolate(lidar_bev_processed,
                                                        size=cam_bev_processed.shape[2:],
                                                        mode='bilinear',
                                                        align_corners=False)
            # assert torch.all(torch.isfinite(lidar_bev_processed_resized)
            #                  ), "NaN/Inf detected after F.interpolate on lidar_bev"
        else:
            lidar_bev_processed_resized = lidar_bev_processed  # No resizing needed

        fused_features = torch.cat(
            [cam_bev_processed, lidar_bev_processed_resized], dim=1)
        # assert torch.all(torch.isfinite(fused_features)
        #                  ), "NaN/Inf detected after torch.cat"

        # 5. Final processing
        output = self.fusion_conv(fused_features)
        # assert torch.all(torch.isfinite(
        #     output)), "!!! NaN/Inf detected in final output of FusionNet.forward !!!"

        return output, depth_prob  # 返回最终输出和深度概率


def compile_model_fusion(grid_conf, data_aug_conf, outC, lidar_inC=1):
    """
    创建并返回一个FusionNet实例，用于摄像头和LiDAR融合的BEV表示。

    Args:
        grid_conf: 网格配置参数
        data_aug_conf: 数据增强配置参数
        outC: 输出通道数
        lidar_inC: LiDAR输入通道数，默认为1

    Returns:
        FusionNet对象实例
    """
    return FusionNet(grid_conf, data_aug_conf, outC, lidar_inC=lidar_inC)


# 为了保持向后兼容性，保留原名称的函数接口
def compile_model(grid_conf, data_aug_conf, outC, lidar_inC=1):
    """
    创建并返回一个模型实例。根据是否提供lidar_inC参数来决定返回FusionNet还是LiftSplatShoot。

    Args:
        grid_conf: 网格配置参数
        data_aug_conf: 数据增强配置参数
        outC: 输出通道数
        lidar_inC: LiDAR输入通道数（如果不为None，则返回FusionNet）

    Returns:
        FusionNet或LiftSplatShoot模型实例
    """
    if lidar_inC is not None:
        # 返回多模态融合模型
        return FusionNet(grid_conf, data_aug_conf, outC, lidar_inC=lidar_inC)
    else:
        # 返回原始LSS模型
        return LiftSplatShoot(grid_conf, data_aug_conf, outC)


class DepthRefinementNetV2(nn.Module):
    """
    改进版深度细化网络，用于细化深度预测。
    接收初始深度logits、log方差和FPN特征作为输入。
    输出细化后的深度logits和log方差。
    """

    def __init__(self, D, fpn_channels, intermediate_channels=128):
        super().__init__()
        self.D = D
        # Input channels: D (logits) + D (log_var) + fpn_channels (p3)
        input_channels = 2 * D + fpn_channels
        self.conv_block = nn.Sequential(
            nn.Conv2d(input_channels, intermediate_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(intermediate_channels, intermediate_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(inplace=True),
            # Output refined logits and log_variance (2 * D channels)
            nn.Conv2d(intermediate_channels, 2 * D, kernel_size=1, padding=0)
        )

    def forward(self, initial_logits, initial_log_var, fpn_feat):
        # Concatenate inputs along the channel dimension
        x = torch.cat([initial_logits, initial_log_var, fpn_feat], dim=1)
        # Pass through convolutional block
        refined_output = self.conv_block(x)
        # Split into refined logits and log variance
        refined_logits = refined_output[:, :self.D]
        refined_log_var = refined_output[:, self.D:]
        return refined_logits, refined_log_var
