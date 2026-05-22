# TopoSPADE

基于 **pix2pixHD** 框架的地形灰度图生成模型，在标准 GAN + 多尺度判别器 + 特征匹配 + VGG 感知损失之上，扩展结构感知编码、SPADE 条件调制与高度一致性损失。

---

## 实验环境

| 项目 | 配置 |
|------|------|
| 操作系统 | Windows 11 |
| CPU | 12th Gen Intel(R) Core(TM) i5-12400F @ 2.50 GHz |
| 内存 | 32 GB |
| GPU | NVIDIA GeForce RTX 4060（8 GB 显存） |
| 语言 | Python 3.12 |
| 深度学习框架 | PyTorch 2.0.1 |

```bash
pip install -r requirements.txt
```

---

## 数据与预处理

- 地形样本在**数据准备阶段**预处理至适合网络的尺寸；训练集与测试集按约定比例划分（本仓库不包含数据集，需自行准备）。
- 数据根目录：`datasets/TopoSPADE/`
- pix2pixHD 对齐格式：`train_A` / `train_B` / `train_inst`（测试对应 `test_*`）
- **输入缩放（默认）**：按宽度缩放到 **1024 像素**，保持纵横比（`--resize_or_crop scale_width --loadSize 1024`），不做固定正方形裁剪。

---

## 数据增强（仅训练阶段）

为增强泛化并抑制过拟合，训练时对样本随机施加：

- 水平翻转
- 垂直翻转
- 小角度旋转（默认最大 ±5°，`--rotate_deg 5`）

测试阶段不启用上述增强。可用 `--no_flip` 关闭全部翻转与旋转增强。

---



### 训练命令

```bash
python train.py --label_nc 0 --input_nc 4 --output_nc 1 --name TopoSPADE --dataroot ./datasets/TopoSPADE --resize_or_crop scale_width --loadSize 1024 --batchSize 1 --niter 100 --niter_decay 100 --lr 0.0002 --beta1 0.5 --beta2 0.999 --lambda_feat 10.0 --height_loss_weight 10.0 --use_structure_aware
```

权重保存在 `checkpoints/TopoSPADE/`。

---

## 测试 / 推理

```bash
python test.py --name TopoSPADE --dataroot ./datasets/TopoSPADE --use_structure_aware --which_epoch latest
```

---

## 损失函数说明（与论文表述对应）

1. **GAN 损失** + **多尺度判别器特征匹配**（权重 λ_feat=10）+ **VGG 感知损失**（同样乘以 λ_feat=10，见 `models/pix2pixHD_model.py`）。
2. **高度一致性损失**（`--height_loss_weight`，默认 10）：
   - 生成图与真实图的像素 **L₁**；
   - 当输入标签含第 4 通道（高度编码）时，额外约束生成图与输入高度通道的 **Sobel 梯度** 一致性（梯度项权重 0.3，与 L₁ 相加后再乘 `height_loss_weight`）。

---

## 目录结构

```
daimafenxiang/
├── train.py
├── test.py
├── models/
├── options/
├── data/
└── util/
```

## 未包含

- 数据集、划分脚本、checkpoint、测试结果
- 消融实验与 ArcGIS 等辅助工具
