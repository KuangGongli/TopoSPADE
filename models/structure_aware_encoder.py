import torch
import torch.nn as nn
import torch.nn.functional as F

class StructureAwareEncoder(nn.Module):
    """
    结构感知编码器 - 保留地形草图的拓扑结构信息
    特点：
    1. 使用跳跃连接保留空间拓扑结构
    2. 多尺度特征提取
    3. 结构显著区域掩膜特征提取
    """
    def __init__(self, input_nc=3, ngf=64, n_downsampling=4, norm_layer=nn.BatchNorm2d):
        super(StructureAwareEncoder, self).__init__()
        
        self.n_downsampling = n_downsampling
        activation = nn.ReLU(True)
        
        # 初始卷积层 - 提取基础特征
        self.initial_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            norm_layer(ngf),
            activation
        )
        
        # 多尺度编码器层
        self.encoders = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        
        for i in range(n_downsampling):
            mult = 2 ** i
            # 编码器层
            encoder = nn.Sequential(
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                norm_layer(ngf * mult * 2),
                activation
            )
            self.encoders.append(encoder)
            
            # 跳跃连接 - 保留结构信息
            skip_conv = nn.Sequential(
                nn.Conv2d(ngf * mult, ngf * mult, kernel_size=1),
                norm_layer(ngf * mult),
                activation
            )
            self.skip_connections.append(skip_conv)
        
        # 结构特征融合层
        self.structure_fusion = nn.ModuleList()
        for i in range(n_downsampling):
            mult = 2 ** i
            fusion = nn.Sequential(
                nn.Conv2d(ngf * mult * 2, ngf * mult, kernel_size=3, padding=1),
                norm_layer(ngf * mult),
                activation
            )
            self.structure_fusion.append(fusion)
    
    def forward(self, x):
        """
        前向传播
        输入: x - 地形草图 (B, 3, H, W)
        输出: encoded_features - 编码特征列表
              skip_features - 跳跃连接特征列表
        """
        # 初始特征提取
        features = self.initial_conv(x)
        
        # 多尺度编码 + 跳跃连接
        encoded_features = []
        skip_features = []
        
        current_feature = features
        for i, (encoder, skip_conv) in enumerate(zip(self.encoders, self.skip_connections)):
            # 保存跳跃连接特征（保留结构信息）
            skip_feature = skip_conv(current_feature)
            skip_features.append(skip_feature)
            
            # 编码
            current_feature = encoder(current_feature)
            encoded_features.append(current_feature)
        
        return encoded_features, skip_features

class StructureAwareDecoder(nn.Module):
    """
    结构感知解码器 - 利用跳跃连接重建结构信息
    """
    def __init__(self, ngf=64, n_downsampling=4, output_nc=3, norm_layer=nn.BatchNorm2d):
        super(StructureAwareDecoder, self).__init__()
        
        activation = nn.ReLU(True)
        
        # 解码器层
        self.decoders = nn.ModuleList()
        self.structure_fusion = nn.ModuleList()
        
        for i in range(n_downsampling):
            # 解码器通道计算：从最深层开始，逐层减少通道数
            # 输入：encoded_features[-1] 是 1024 通道
            # i=0: 1024->512, i=1: 512->256, i=2: 256->128, i=3: 128->64
            if i == 0:
                # 第一层：从最深层开始
                input_channels = ngf * (2 ** n_downsampling)  # 1024
                output_channels = ngf * (2 ** (n_downsampling - 1))  # 512
            else:
                # 后续层：逐层减少
                input_channels = ngf * (2 ** (n_downsampling - i))  # 512, 256, 128
                output_channels = ngf * (2 ** (n_downsampling - i - 1)) if i < n_downsampling - 1 else ngf  # 256, 128, 64
            
            # 解码器层
            decoder = nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, 
                                 kernel_size=3, stride=2, padding=1, output_padding=1),
                norm_layer(output_channels),
                activation
            )
            self.decoders.append(decoder)
            
            # 结构特征融合 - 融合解码后的特征和跳跃连接特征
            fusion_input_channels = output_channels + output_channels  # 解码特征 + 跳跃连接特征
            fusion = nn.Sequential(
                nn.Conv2d(fusion_input_channels, output_channels, kernel_size=3, padding=1),
                norm_layer(output_channels),
                activation
            )
            self.structure_fusion.append(fusion)
        
        # 最终输出层
        self.final_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        )
    
    def forward(self, encoded_features, skip_features):
        """
        前向传播
        输入: encoded_features - 编码特征列表
              skip_features - 跳跃连接特征列表
        输出: 重建的图像
        """
        # 从最深层开始解码
        current_feature = encoded_features[-1]
        
        # 逐层解码并融合跳跃连接特征
        for i, (decoder, fusion) in enumerate(zip(self.decoders, self.structure_fusion)):
            # 解码
            current_feature = decoder(current_feature)
            
            # 融合跳跃连接特征（如果存在）
            if i < len(skip_features):
                skip_idx = len(skip_features) - 1 - i
                skip_feature = skip_features[skip_idx]
                
                # 调整尺寸匹配
                if skip_feature.size()[2:] != current_feature.size()[2:]:
                    skip_feature = F.interpolate(skip_feature, size=current_feature.size()[2:], mode='bilinear', align_corners=False)
                
                # 调整通道数匹配
                if skip_feature.size(1) != current_feature.size(1):
                    # 使用1x1卷积调整通道数
                    if not hasattr(self, f'channel_adapter_{i}'):
                        adapter = nn.Conv2d(skip_feature.size(1), current_feature.size(1), kernel_size=1).to(skip_feature.device)
                        setattr(self, f'channel_adapter_{i}', adapter)
                    else:
                        adapter = getattr(self, f'channel_adapter_{i}')
                    skip_feature = adapter(skip_feature)
                
                # 特征融合
                fused_feature = torch.cat([current_feature, skip_feature], dim=1)
                current_feature = fusion(fused_feature)
        
        # 最终输出
        output = self.final_conv(current_feature)
        return output

class StructureAwareGenerator(nn.Module):
    """
    结构感知生成器 - 结合编码器和解码器
    """
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_downsampling=4, norm_layer=nn.BatchNorm2d):
        super(StructureAwareGenerator, self).__init__()
        
        self.encoder = StructureAwareEncoder(input_nc, ngf, n_downsampling, norm_layer)
        self.decoder = StructureAwareDecoder(ngf, n_downsampling, output_nc, norm_layer)
    
    def forward(self, x):
        """
        前向传播
        输入: x - 地形草图 (B, 3, H, W)
        输出: 生成的地形图 (B, 3, H, W)
        """
        # 编码
        encoded_features, skip_features = self.encoder(x)
        
        # 解码
        output = self.decoder(encoded_features, skip_features)
        
        return output

