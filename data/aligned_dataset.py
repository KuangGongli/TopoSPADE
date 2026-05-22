import os.path
import torch
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        ### input A (label maps)
        dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        ### input B (real images)
        if opt.isTrain or opt.use_encoded_image:       #在测试时需要加入--use_encoded_image添加原图进行对比
            dir_B = '_B' if self.opt.label_nc == 0 else '_img'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)  
            self.B_paths = sorted(make_dataset(self.dir_B))

        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:                              
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        self.dataset_size = len(self.A_paths) 
      
    def __getitem__(self, index):        
        ### input A (label maps)
        A_path = self.A_paths[index]              
        A = Image.open(A_path)        
        params = get_params(self.opt, A.size)
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params)
            A_tensor = transform_A(A.convert('RGB'))  # [-1,1]

            # 方案1：基于色带查找表的绝对高度映射
            # 根据valley.clr和ridge.clr色带的RGB值规律建立颜色到绝对高度的映射
            # - ridge.clr（红色带）: R从255→130，高度值从0→47（低→高），R值越大高度越高
            # - valley.clr（绿色带）: G从130→255，高度值从0→47（低→高），G值越大高度越低（山谷越深）
            R = A_tensor[0:1, :, :]  # [-1, 1]
            G = A_tensor[1:2, :, :]  # [-1, 1]
            B = A_tensor[2:3, :, :]  # [-1, 1]
            
            # 将RGB从[-1,1]转换到[0,255]范围（用于色带查找）
            R_uint8 = ((R + 1.0) * 127.5).clamp(0, 255)
            G_uint8 = ((G + 1.0) * 127.5).clamp(0, 255)
            B_uint8 = ((B + 1.0) * 127.5).clamp(0, 255)
            
            # 判断是红色带（山脊）还是绿色带（山谷）
            # 红色带：R明显大于G和B
            # 绿色带：G明显大于R和B
            is_ridge = (R_uint8 > G_uint8 + 10) & (R_uint8 > B_uint8 + 10)  # 红色占主导
            is_valley = (G_uint8 > R_uint8 + 10) & (G_uint8 > B_uint8 + 10)  # 绿色占主导
            is_neutral = ~(is_ridge | is_valley)  # 中性区域（白色、黑色、过渡色）
            
            # 初始化高度通道（归一化到[0,1]，后续会转换到[-1,1]）
            height_normalized = torch.zeros_like(R)
            
            # 红色带（山脊）：根据ridge.clr规律，R从255→130，高度从低→高
            # 在ridge.clr中：值1对应R=255（最低），值46对应R=130（最高）
            # 所以：R值越小（越深红）→ 高度越高
            if is_ridge.any():
                # 将R值映射到高度：R在[130, 255]范围内，R越小高度越高
                R_ridge = R_uint8[is_ridge]
                # 归一化R值：将[130, 255]映射到[0, 1]，注意：R越小，归一化值越小，但高度越高
                # 所以需要反转：R_norm = (255 - R) / (255 - 130)
                R_norm = (255.0 - R_ridge) / (255.0 - 130.0)
                R_norm = R_norm.clamp(0, 1)
                # 映射到高海拔范围[0.5, 1.0]：R越小（R_norm越大），高度越高
                height_normalized[is_ridge] = R_norm * 0.5 + 0.5
            
            # 绿色带（山谷）：根据valley.clr规律，G从130→255，高度从低→高
            # 在valley.clr中：值1对应G=130（最低），值46对应G=255（最高）
            # 但用户说"绿色越深越低"，所以：G值越小（越深绿）→ 高度越低
            if is_valley.any():
                # 将G值映射到高度：G在[130, 255]范围内，G越小高度越低
                G_valley = G_uint8[is_valley]
                # 归一化G值：将[130, 255]映射到[0, 1]，注意：G越小，归一化值越小，高度越低
                # 所以：G_norm = (G - 130) / (255 - 130)，G越小，G_norm越小
                G_norm = (G_valley - 130.0) / (255.0 - 130.0)
                G_norm = G_norm.clamp(0, 1)
                # 映射到低海拔范围[0, 0.5]：G越小（G_norm越小），高度越低
                height_normalized[is_valley] = (1.0 - G_norm) * 0.5
            
            # 中性区域（白色、黑色、过渡色）：使用R-G差分作为相对高度
            if is_neutral.any():
                # 对于中性区域，使用简单的R-G差分
                height_diff = 0.5 * (R[is_neutral] - G[is_neutral])
                # 将差分从[-1, 1]映射到[0, 1]，然后映射到中等高度范围[0.25, 0.75]
                height_diff_norm = (height_diff + 1.0) * 0.5  # [0, 1]
                height_normalized[is_neutral] = height_diff_norm * 0.5 + 0.25
            
            # 转换回[-1, 1]范围，使其与真实灰度图的归一化范围一致
            # 真实灰度图：深灰色（低高度）→ 接近-1，浅灰色（高高度）→ 接近1
            height = height_normalized * 2.0 - 1.0
            
            # 确保高度通道在合理范围内
            height = torch.clamp(height, -1.0, 1.0)
            
            A_tensor = torch.cat([A_tensor, height], dim=0)  # [4, H, W]
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A_tensor = transform_A(A) * 255.0

        B_tensor = inst_tensor = feat_tensor = 0
        ### input B (real images)
        if self.opt.isTrain or self.opt.use_encoded_image:
            B_path = self.B_paths[index]   
            B = Image.open(B_path).convert('RGB')
            transform_B = get_transform(self.opt, params)      
            B_tensor = transform_B(B)

        ### if using instance maps        
        if not self.opt.no_instance:
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform_A(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]            
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_A(feat))                            

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 
                      'feat': feat_tensor, 'path': A_path}

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'