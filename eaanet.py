import torch
from torch import nn
import torch.nn.functional as F
import os, sys
sys.path.append("G:\\segyb")#测试的时候需要添加项目根目录，使用Pycharm则无需添加，Pycharm会自动添加项目根目录
from models.seg_models.mamba_moudle import VSSLayer


class Convstem(nn.Module): #卷积干：负责初步特征提取以及下采样，降低分辨率
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            # 第一层下采样
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第二层下采样
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    def forward(self, x):
        return self.layer(x)

class MCAM(nn.Module):#多尺度通道自适应模块MCAM
    def __init__(self, in_channels):
        super().__init__()
        
        # 内部公共卷积块
        def conv_bn_relu(in_ch, out_ch, kernel, padding, dilation=1):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel, padding=padding, dilation=dilation, 
                          bias=False, padding_mode="reflect"),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            )

        # 1. 三个不同感受野的分支
        self.branch1 = conv_bn_relu(in_channels, in_channels, 3, padding=1)
        self.branch2 = conv_bn_relu(in_channels, in_channels, 3, padding=2, dilation=2)
        self.branch3 = conv_bn_relu(in_channels, in_channels, 3, padding=3, dilation=3)

        # 2. 动态选择模块 (基于 Concat)
        # 拼接后的总通道数为 in_channels * 3
        total_concat_channels = in_channels * 3
        # 确保 reduction 至少为 8，防止通道数过小时 linear 层失效
        reduction = max(total_concat_channels // 6, 8) 
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(total_concat_channels, reduction),
            nn.LeakyReLU(0.2, inplace=True),
            # 最终输出仍为 in_channels * 3，用于给 3 个分支分配权重
            nn.Linear(reduction, in_channels * 3) 
        )

    def forward(self, x):
        b, c, h, w = x.size()
        
        # 提取各分支特征
        f1 = self.branch1(x)
        f2 = self.branch2(x)
        f3 = self.branch3(x)

        # --- 核心修改：从 Sum 改为 Concat ---
        # 拼接特征图，形状为 (b, 3*c, h, w)
        f_concat = torch.cat([f1, f2, f3], dim=1) 

        # 全局平均池化得到特征向量 (b, 3*c)
        stat = self.gap(f_concat).view(b, -1)
        
        # 通过全连接层生成权重，并重塑为 (b, 3, c) 方便 Softmax
        # 这里的 3 代表三个分支，c 代表每个通道都有独立的权重
        weights = self.fc(stat).view(b, 3, c)
        
        # 在分支维度（dim=1）做 Softmax，确保三个分支对应通道的权重之和为 1
        weights = F.softmax(weights, dim=1) 
        
        # 提取权重并扩展维度用于按通道相乘
        w1 = weights[:, 0, :].view(b, c, 1, 1)
        w2 = weights[:, 1, :].view(b, c, 1, 1)
        w3 = weights[:, 2, :].view(b, c, 1, 1)

        # 加权融合（这里依然通过加权求和输出，但权重是基于拼接特征计算的，更精准）
        out = f1 * w1 + f2 * w2 + f3 * w3
        return out

def Img2Seq(x):
    # (B, C, H, W) -> (B, L, C)
    b, c, _, _ = x.size()
    x = x.view(b, c, -1)
    x = x.permute(0, 2, 1).contiguous()
    return x

def Seq2Img(x, h, w):
    # (B, L, C) -> (B, C, H, W)
    b, l, c = x.shape
    if h * w != l:
        raise ValueError(f"Shape mismatch: {h}x{w} != {l}")
    return x.permute(0, 2, 1).contiguous().view(b, c, h, w)

class FFN(nn.Module):
    def __init__(self, embed_dim, expansion_factor=6, dropout=0.1):
        super().__init__()
        hidden_dim = embed_dim * expansion_factor
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ==========================================
# 2. Swin Transformer 核心组件 (新增部分)
# ==========================================

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        H, W: Image height and width
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    r""" 基于窗口的多头自注意力 (W-MSA) + 相对位置偏置
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 定义相对位置偏置表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )

        # 生成相对位置索引
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        # x: [num_windows*B, N, C]
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B_, num_heads, N, C//num_heads]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # 添加相对位置偏置
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=6., drop=0., attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size # 新增：移位大小

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=True, attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FFN(embed_dim=dim, expansion_factor=int(mlp_ratio), dropout=drop)

    def forward(self, x, H, W):
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # --- [Step 1] Padding 处理 ---
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x.shape

        # --- [Step 2] Cyclic Shift (循环移位) ---
        if self.shift_size > 0:
            # 向左上角移动，使得原本边缘的像素进入窗口中心进行交互
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # --- [Step 3] Window Partition ---
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # --- [Step 4] W-MSA / SW-MSA ---
        # 注意：严格来说 SW-MSA 需要一个 Mask 来防止移位后的非相邻像素互相看
        # 但在特征提取阶段，即使不加 Mask，效果通常也比不移位好得多
        attn_windows = self.attn(x_windows)

        # --- [Step 5] Window Reverse & Reverse Shift ---
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        if self.shift_size > 0:
            # 移回来
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # --- [Step 6] 切除 Padding 并做残差 ---
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = shortcut + x
        x = x + self.ffn(self.norm2(x))
        return x

# ==========================================
# 3. 修改后的主模块 (Efficient_Hybird_Extractor)
# ==========================================

class Efficient_Hybrid_Extractor(nn.Module):
    def __init__(self, in_channels, nums_head, trans_block_nums, window_size=7):
        super().__init__()
        self.MSCA = MCAM(in_channels)
        
        # 修改点：根据索引 i 来决定是否移位
        self.blocks = nn.ModuleList([
            SwinBlock(
                dim=in_channels, 
                num_heads=nums_head, 
                window_size=window_size,
                # i 为奇数时移位，偶数时不移位
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=6, 
                drop=0.1, 
                attn_drop=0.1
            )
            for i in range(trans_block_nums)
        ])
        
    def forward(self, x):
        _, _, h, w = x.shape

        y = x
        
        # [Step 1] 卷积空间增强 (Input: B, C, H, W)
        x = self.MSCA(x)
        
        # [Step 2] 转换维度 (Output: B, L, C)
        x = Img2Seq(x)
        
        # [Step 3] 逐层通过 Swin Block
        # 关键修改：我们需要把 h, w 传进去，以便 Block 内部做窗口切分
        for block in self.blocks:
            x = block(x, h, w)
            
        # [Step 4] 还原回图像维度 (Output: B, C, H, W)
        x = Seq2Img(x, h, w)
        x = x + y
        
        return x

#定义mamba特征提取模块
class Efficient_Hybrid_Extractor_Mamba(nn.Module):
    def __init__(self, in_channels, vss_block_num=2): # 修正拼写
        super().__init__()
        self.MCAM = MCAM(in_channels)
        self.VSS = VSSLayer(dim=in_channels, depth=vss_block_num)
    
    def forward(self, x):
        identity = x # 使用 identity 命名残差路径是通用惯例
        
        # 1. 卷积空间特征提取
        x = self.MCAM(x) 
        
        # 2. 维度转换以适配 VSSLayer (B, C, H, W) -> (B, H, W, C)
        x = x.permute(0, 2, 3, 1).contiguous() 
        
        # 3. Mamba 序列建模
        x = self.VSS(x)
        
        # 4. 维度还原 (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2).contiguous()
        
        # 5. 残差连接
        return x + identity





#多尺度自适应融合模块，高级语义信息特征图需要没有上采样的，inchannels只需要单个特征图的
class Adaptive_Multiscale_Fusion(nn.Module):#moudle2
    def __init__(self, in_channels):
        super().__init__()

        # 进阶点：输入通道变为 2 * in_channels
        # 这样卷积核可以同时看到语义特征和细节特征的差异
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1) # 最终压缩为 1 个通道
        )
        self.active = nn.Sigmoid()

    def forward(self, f_high, f_low):
        """
        f_high: 高级语义特征 (B, C, H/2, W/2)
        f_low:  低级细节特征 (B, C, H, W)
        """
        # 1. 对高级特征进行 2 倍双线性插值上采样
        f_high_up = F.interpolate(f_high, scale_factor=2, mode='bilinear', align_corners=False)

        # 2. 进阶融合逻辑：特征拼接 (Concat)
        # 将上采样后的语义特征与细节特征在通道维度合并
        f_concat = torch.cat([f_high_up, f_low], dim=1) # (B, 2C, H, W)

        # 3. 生成空间权重地图 
        # 卷积核现在可以对比两个特征图：哪里需要补语义，哪里需要留细节
        a = self.conv(f_concat)
        a = self.active(a) # (B, 1, H, W)

        # 4. 互补加权融合
        # a 接近 1 时倾向 f_high_up，a 接近 0 时倾向 f_low
        out = a * f_high_up + (1 - a) * f_low
        
        return out


class ChannelAttention(nn.Module):#通道注意力机制
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SpatialAttention(nn.Module):#空间注意力机制
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 在通道维度做最大池化和平均池化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)

def shuffle_chnls(x, groups=3):
    """Channel Shuffle"""

    bs, chnls, h, w = x.data.size()

    # 如果通道数不是分组的整数被，则无法进行Channel Shuffle操作，直接返回x
    if chnls % groups:
        raise AttributeError('Please confirm channels can be exact division!')

    # 计算用于Channel Shuffle的一个group的的通道数
    chnls_per_group = chnls // groups

    # 执行channel shuffle操作，不要直接用view变成5个维度，导出的onnx会报错
    x = x.unsqueeze(1)#在通道维度加个维度，告诉算子管理通道，他就指导分通道了
    x = x.view(bs, groups, chnls_per_group, h, w)  # 将通道那个维度拆分为 (g,n) 两个维度
    x = torch.transpose(x, 1, 2).contiguous()  # 将这两个维度转置变成 (n,g)
    x = x.view(bs, chnls, h, w)  # 最后重新reshape成一个维度 g × n g\times ng×n

    return x

class Attention_Enhancement_Module(nn.Module):#moudle3
    def __init__(self, in_channels, out_channels, groups=4): # 建议 groups 不要太大，4 是个平衡点
        super().__init__()
        self.groups = groups
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()

        combined_channels = in_channels * 3
        
        # 建议方案：分组卷积提特征 + 1x1 卷积全局降维
        self.fusion_conv = nn.Sequential(
            # 1. 分组卷积层：保持 3C 通道，负责 3x3 空间特征提取
            nn.Conv2d(combined_channels, in_channels, kernel_size=3, padding=1, groups=groups, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.01, inplace=True), # 中间层加激活，增强非线性
        )
        # self.conv_3x3 = nn.Sequential(
        #     # 2. 3x3 卷积层：负责提取融合后的特征
        #     nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode="reflect", bias=False),
        #     nn.BatchNorm2d(out_channels),
        #     nn.LeakyReLU(0.01, inplace=True)
        # )
        self.conv_1x1 = nn.Sequential(
            # 2. 3x3 卷积层：负责提取融合后的特征
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding_mode="reflect", bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.01, inplace=True)
        )
    def forward(self, x):
        out = torch.cat([x, self.ca(x), self.sa(x)], dim=1)
        out = shuffle_chnls(out, 3)
        out = self.fusion_conv(out)
        out = out + x
        out = self.conv_1x1(out)
        return out




class PatchMerging(nn.Module):#4.Patch Merging模块
    def __init__(self, embed_dim, out_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.reduction = nn.Linear(4 * embed_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(4 * embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0, f"Height ({H}) and width ({W}) must be even."
        # Reshape and select patches
        x0 = x[:, :, 0::2, 0::2]  # B, C, H/2, W/2
        x1 = x[:, :, 1::2, 0::2]  # B, C, H/2, W/2
        x2 = x[:, :, 0::2, 1::2]  # B, C, H/2, W/2
        x3 = x[:, :, 1::2, 1::2]  # B, C, H/2, W/2

        # Concatenate along the channel dimension
        x = torch.cat([x0, x1, x2, x3], 1)  # B, 4*C, H/2, W/2

        # Reshape to merge the channels
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(B, -1, 4 * C)  # B, H/2*W/2, 4*C
        x = self.norm(x)  # Normalization
        x = self.reduction(x)  # Channel reduction

        # Reshape back to (B, Dim, H/2, W/2)
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(B, -1, H // 2, W // 2)
        return x



#EMA-Net
class EAA_Net(nn.Module):
    def __init__(self, in_channels, num_classes, mamba=False):
        super().__init__()

        self.convstem = Convstem(in_channels, 16)
        
        if mamba:
            self.EHE_1 = Efficient_Hybrid_Extractor_Mamba(in_channels=16, vss_block_num=2)
            self.EHE_2 = Efficient_Hybrid_Extractor_Mamba(in_channels=32, vss_block_num=2)
            self.EHE_3 = Efficient_Hybrid_Extractor_Mamba(in_channels=32, vss_block_num=4)
        else:
            self.EHE_1 = Efficient_Hybrid_Extractor(in_channels=16, nums_head=4, trans_block_nums=2)
            self.EHE_2 = Efficient_Hybrid_Extractor(in_channels=32, nums_head=4, trans_block_nums=2)
            self.EHE_3 = Efficient_Hybrid_Extractor(in_channels=32, nums_head=4, trans_block_nums=4)

        self.PatchMerging_1 = PatchMerging(16, 32)
        self.PatchMerging_2 = PatchMerging(32, 32)

        self.MSAF_1 = Adaptive_Multiscale_Fusion(in_channels=32)
        self.MSAF_2 = Adaptive_Multiscale_Fusion(in_channels=16)

        self.AEM_1 = Attention_Enhancement_Module(in_channels=32, out_channels=16, groups=8)
        self.AEM_2 = Attention_Enhancement_Module(in_channels=16, out_channels=16, groups=4)

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, padding_mode="reflect", bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=num_classes, kernel_size=1, bias=False)
    
    def forward(self, feature_map):
        x = self.convstem(feature_map)#(16,H/4,W/4)

        feature_map_1 = self.EHE_1(x)#(16,H/4,W/4)

        feature_map_down_1 = self.PatchMerging_1(feature_map_1)#(32,H/8,W/8)
        #feature_map_down_1 = F.max_pool2d(feature_map_1, kernel_size=2, stride=2)#(16,H/8,W/8)

        feature_map_2 = self.EHE_2(feature_map_down_1)#(32,H/8,W/8)

        feature_map_down_2 = self.PatchMerging_2(feature_map_2)
        #feature_map_down_2 = F.max_pool2d(feature_map_2, kernel_size=2, stride=2)#(32,H/16,W/16)

        feature_map_3 = self.EHE_3(feature_map_down_2)#(32,H/16,W/16)

        fusion_1 = self.MSAF_1(feature_map_3, feature_map_2)#(32,H/8,W/8)

        enhance_1 = self.AEM_1(fusion_1)#(16,H/8,W/8)

        fusion_2 = self.MSAF_2(enhance_1, feature_map_1)#(16,H/4,W/4)

        enhance_2 = self.AEM_2(fusion_2)#(16,H/4,W/4)

        out = F.interpolate(enhance_2, scale_factor=4,  mode='bilinear', align_corners=False)#(16,H,W)

        out = self.conv3(out)

        return self.conv1(out)



if __name__ == "__main__":
    # 1. 检查 GPU 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 正在使用设备: {device} ---")

    # 2. 初始化模型并搬运到 GPU
    # 注意：如果 mamba=True，必须确保你的 selective_scan_cuda 已安装
    model = EAA_Net(in_channels=3, num_classes=4, mamba=True).to(device)
    
    # 3. 构造输入数据并搬运到 GPU
    x = torch.randn(4, 3, 224, 224).to(device)

    # 4. 执行前向传播
    # 使用 torch.no_grad() 可以在测试时节省显存，加快速度
    with torch.no_grad():
        try:
            y = model(x)
            print(f"输出形状: {y.shape}")
            print("GPU 测试成功！")
        except Exception as e:
            print(f"GPU 测试失败，错误信息: {e}")


    
        







    

