import torch
import torch.nn as nn
import torch.nn.functional as F

class StructureAwareSPADE(nn.Module):
    """
    结构感知SPADE模块 - 结合结构特征进行空间自适应归一化
    """
    def __init__(self, norm_nc, label_nc, structure_nc=0):
        super().__init__()
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        
        # 结构感知的MLP
        nhidden = 128
        
        # MLP for semantic map
        self.mlp_semantic = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # MLP for structure features (if provided)
        if structure_nc > 0:
            self.mlp_structure = nn.Sequential(
                nn.Conv2d(structure_nc, nhidden, kernel_size=3, padding=1),
                nn.ReLU()
            )
            # Combined MLP for gamma and beta
            self.mlp_gamma = nn.Conv2d(nhidden * 2, norm_nc, kernel_size=3, padding=1)
            self.mlp_beta = nn.Conv2d(nhidden * 2, norm_nc, kernel_size=3, padding=1)
        else:
            self.mlp_structure = None
            # Only semantic MLP for gamma and beta
            self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
            self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        
        # 结构特征权重
        self.structure_weight = nn.Parameter(torch.ones(1) * 0.5)
    
    def forward(self, x, segmap, structure_features=None):
        """
        前向传播
        输入: x - 输入特征
              segmap - 地形草图
              structure_features - 结构特征（可选）
        """
        normalized = self.param_free_norm(x)
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        
        # 处理语义特征
        actv_semantic = self.mlp_semantic(segmap)
        
        # 处理结构特征
        if structure_features is not None and self.mlp_structure is not None:
            structure_features = F.interpolate(structure_features, size=x.size()[2:], mode='bilinear', align_corners=False)
            actv_structure = self.mlp_structure(structure_features)
            # 融合语义和结构激活
            actv_combined = torch.cat([actv_semantic, actv_structure], dim=1)
        else:
            actv_combined = actv_semantic
            
        gamma = self.mlp_gamma(actv_combined)
        beta = self.mlp_beta(actv_combined)
        
        return normalized * (1 + gamma) + beta

class StructureAwareSPADEResnetBlock(nn.Module):
    """
    结构感知SPADE残差块
    """
    def __init__(self, fin, fout, label_nc, structure_nc=0):
        super().__init__()
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # structure-aware spade normalization layers
        self.spade_0 = StructureAwareSPADE(fin, label_nc, structure_nc)
        self.spade_1 = StructureAwareSPADE(fmiddle, label_nc, structure_nc)
        if self.learned_shortcut:
            self.spade_s = StructureAwareSPADE(fin, label_nc, structure_nc)

    def forward(self, x, segmap, structure_features=None):
        x_s = self.shortcut(x, segmap, structure_features)

        dx = self.spade_0(x, segmap, structure_features)
        dx = F.relu(dx)
        dx = self.conv_0(dx)

        dx = self.spade_1(dx, segmap, structure_features)
        dx = F.relu(dx)
        dx = self.conv_1(dx)

        out = x_s + dx
        return out

    def shortcut(self, x, segmap, structure_features=None):
        if self.learned_shortcut:
            x_s = self.spade_s(x, segmap, structure_features)
            x_s = self.conv_s(x_s)
        else:
            x_s = x
        return x_s


