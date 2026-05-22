import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class StructureLoss(nn.Module):
    """
    结构感知损失函数 - 保持地形结构的连续性
    """
    def __init__(self, weight=1.0):
        super(StructureLoss, self).__init__()
        self.weight = weight
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, pred, target):
        """
        计算结构损失
        输入: pred - 预测图像
              target - 真实图像
        """
        # 梯度损失 - 保持边缘结构
        grad_loss = self.gradient_loss(pred, target)
        
        # 结构相似性损失
        structure_loss = self.structure_similarity_loss(pred, target)
        
        # 总损失
        total_loss = grad_loss + structure_loss
        
        return total_loss * self.weight
    
    def gradient_loss(self, pred, target):
        """
        梯度损失 - 保持边缘和结构
        """
        # 转换为灰度图
        if pred.size(1) == 3:
            pred_gray = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
        else:
            pred_gray = pred
            
        if target.size(1) == 3:
            target_gray = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]
        else:
            target_gray = target
        
        # Sobel算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        if pred_gray.is_cuda:
            sobel_x = sobel_x.cuda()
            sobel_y = sobel_y.cuda()
        
        # 计算梯度
        pred_grad_x = F.conv2d(pred_gray, sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred_gray, sobel_y, padding=1)
        target_grad_x = F.conv2d(target_gray, sobel_x, padding=1)
        target_grad_y = F.conv2d(target_gray, sobel_y, padding=1)
        
        # 梯度损失
        grad_loss = self.l1_loss(pred_grad_x, target_grad_x) + self.l1_loss(pred_grad_y, target_grad_y)
        
        return grad_loss
    
    def structure_similarity_loss(self, pred, target):
        """
        结构相似性损失
        """
        # 转换为灰度图
        if pred.size(1) == 3:
            pred_gray = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
        else:
            pred_gray = pred
            
        if target.size(1) == 3:
            target_gray = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]
        else:
            target_gray = target
        
        # 计算结构相似性
        ssim_loss = 1 - self.ssim(pred_gray, target_gray)
        
        return ssim_loss
    
    def ssim(self, x, y):
        """
        简化的结构相似性计算
        """
        mu_x = torch.mean(x)
        mu_y = torch.mean(y)
        
        sigma_x = torch.var(x)
        sigma_y = torch.var(y)
        sigma_xy = torch.mean((x - mu_x) * (y - mu_y))
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2) / \
               ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2))
        
        return ssim

class EdgeLoss(nn.Module):
    """
    边缘损失函数 - 保持地形边缘的清晰度
    """
    def __init__(self, weight=1.0):
        super(EdgeLoss, self).__init__()
        self.weight = weight
        self.l1_loss = nn.L1Loss()
        
    def forward(self, pred, target):
        """
        计算边缘损失
        """
        # 转换为灰度图
        if pred.size(1) == 3:
            pred_gray = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
        else:
            pred_gray = pred
            
        if target.size(1) == 3:
            target_gray = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]
        else:
            target_gray = target
        
        # 使用Laplacian算子检测边缘
        laplacian = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        
        if pred_gray.is_cuda:
            laplacian = laplacian.cuda()
        
        # 计算边缘
        pred_edge = F.conv2d(pred_gray, laplacian, padding=1)
        target_edge = F.conv2d(target_gray, laplacian, padding=1)
        
        # 边缘损失
        edge_loss = self.l1_loss(pred_edge, target_edge)
        
        return edge_loss * self.weight

class TopologyLoss(nn.Module):
    """
    拓扑损失函数 - 保持地形的拓扑结构
    """
    def __init__(self, weight=1.0):
        super(TopologyLoss, self).__init__()
        self.weight = weight
        self.l1_loss = nn.L1Loss()
        
    def forward(self, pred, target):
        """
        计算拓扑损失
        """
        # 转换为灰度图
        if pred.size(1) == 3:
            pred_gray = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
        else:
            pred_gray = pred
            
        if target.size(1) == 3:
            target_gray = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]
        else:
            target_gray = target
        
        # 计算局部极值点
        pred_extrema = self.find_local_extrema(pred_gray)
        target_extrema = self.find_local_extrema(target_gray)
        
        # 拓扑损失
        topology_loss = self.l1_loss(pred_extrema, target_extrema)
        
        return topology_loss * self.weight
    
    def find_local_extrema(self, x):
        """
        找到局部极值点
        """
        # 使用最大池化和最小池化找到局部极值
        max_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        min_pool = -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)
        
        # 极值掩膜
        extrema_mask = torch.abs(max_pool - min_pool)
        
        return extrema_mask


